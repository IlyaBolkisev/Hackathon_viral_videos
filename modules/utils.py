import os

import cv2
import numpy as np
import onnxruntime as rt
from moviepy.editor import VideoFileClip


def warmup_models(path_to_folder):
    models = {}
    for fn in os.listdir(path_to_folder):
        if '.onnx' not in fn:
            continue
        name = fn.split('.')[0]
        models[name] = rt.InferenceSession(f"{path_to_folder}/{fn}", providers=['CUDAExecutionProvider'])
        input_dict = {inp.name: np.random.rand(1, *inp.shape[1:]).astype('float32')
                      for inp in models[name].get_inputs()}
        output_names = [output.name for output in models[name].get_outputs()]
        models[name].run(output_names, input_dict)
    return models


def run_inference(session, input_data):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs


def preprocess_yolo(img, shape_to=(640, 640)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = min(shape_to[0] / img.shape[0], shape_to[1] / img.shape[1])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img_padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    img_padded[:img.shape[0], :img.shape[1]] = img
    img_padded = np.transpose(img_padded, (2, 0, 1)).astype('float32')
    img_padded /= 255
    return img_padded, scale


def preprocess_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=-1)
    return (img / 255).astype('float32')


def calculate_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


def yolo2xyxy(outputs, scales):
    bboxes = []
    confidences = []
    for b in range(outputs.shape[0]):
        output = outputs[b]
        output = np.transpose(output, (1, 0))

        boxes = output[:, :4]
        confs = output[:, 4]

        confidence_threshold = 0.5
        scores = []
        detections = []

        for i in range(len(boxes)):
            if confs[i] > confidence_threshold:
                x, y, w, h = boxes[i]
                x1 = max(x - w / 2, 0)
                y1 = max(y - h / 2, 0)
                x2 = x1 + w
                y2 = y1 + h
                x1, y1, x2, y2 = [round(e / scales[b]) for e in [x1, y1, x2, y2]]
                scores.append(confs[i])
                detections.append([x1, y1, x2, y2])

        ids = cv2.dnn.NMSBoxes(bboxes=detections, scores=scores, score_threshold=0.25, nms_threshold=0.75)
        bboxes.append(np.array(detections)[ids])
        confidences.append(np.array(scores)[ids])
    return bboxes, confidences


def extract_audio_from_video(video_path, audio_output_path):
    video_clip = VideoFileClip(video_path, verbose=False)
    video_clip.audio.write_audiofile(audio_output_path, verbose=False, logger=None)


def process_activity_scores(activity_frames, total_frames):
    result = np.zeros(total_frames)
    result[activity_frames] = 1
    return result


def process_audio_scores(score, original_fps, total_frames):
    fps = int(original_fps)
    result = np.zeros(total_frames)
    score = (np.array(score) * fps).astype(int)
    score = [e - fps//2 + i for e in score for i in range(fps) if e - fps//2 + i < total_frames]
    result[score] = 1
    return result


def process_emotions_scores(emotions_scores, total_frames):
    result = np.zeros(total_frames)
    emotions_id = np.full(total_frames, -1)
    for idx, vec in emotions_scores:
        label = vec.argmax()
        if label != 6 and vec[0][label] > 0.9:
            result[idx] = 1
            emotions_id[idx] = label
    return result, emotions_id

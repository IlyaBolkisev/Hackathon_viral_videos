import cv2
import numpy as np

from modules.utils import preprocess_yolo, run_inference, preprocess_face, yolo2xyxy, calculate_iou
from modules.emotes_expression import draw_emoji

def get_video_features(input_video_path, models, target_fps=10, activity_thresh=0.7):
    cap = cv2.VideoCapture(input_video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = int(original_fps / target_fps) if target_fps < original_fps else 1

    # features
    activity_frames = []
    emotions_scores = []

    bbox_list = []
    processed_frame_numbers = []
    fill_bbox = None
    prev_bbox = None
    prev_frame = None
    frame_number = 0
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            if prev_frame is not None:
                cur_frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
                diff = (cv2.absdiff(cur_frame, prev_frame)).astype(int)
                non_zero_count = np.count_nonzero(diff)
                if non_zero_count > diff.size * activity_thresh:
                    for i in range(frame_skip):
                        if frame_number+i < total_frames:
                            activity_frames.append(frame_number+i)
                prev_frame = cur_frame
            else:
                prev_frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)

            frame, scale = preprocess_yolo(frame_raw)
            outputs = run_inference(models['yolov8n-face'], np.expand_dims(frame, 0))[0]
            xyxy, confidences = yolo2xyxy(outputs, [scale])

            if xyxy[0].shape[0] != 0:
                bbox_sizes = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in xyxy[0]]
                largest_bbox_index = np.argmax(bbox_sizes)

                x1, y1, x2, y2 = xyxy[0][largest_bbox_index]
                faces = [frame_raw[y1:y2, x1:x2]]
                faces = np.stack([preprocess_face(face) for face in faces])
                outputs = run_inference(models['facial_expression'], faces)[0]
                emotions_scores.append((frame_number, outputs))

                curr_bbox = [x1, y1, x2, y2]
                if prev_bbox is None:
                    fill_bbox = curr_bbox
                prev_bbox = curr_bbox
            else:
                curr_bbox = prev_bbox

            processed_frame_numbers.append(frame_number)
            bbox_list.append((frame_number, curr_bbox))
        frame_number += 1

    cap.release()

    return activity_frames, emotions_scores, bbox_list, fill_bbox, total_frames, original_fps


def extrapolate_bboxes(bbox_list, total_frames, fill_bbox):
    all_bboxes = [None] * total_frames
    bbox_index = 0

    bbox_list = [(idx, bbox) if bbox is not None else (idx, fill_bbox) for idx, bbox in bbox_list]

    for i in range(total_frames):
        if bbox_index < len(bbox_list) - 1:
            frame_num1, bbox1 = bbox_list[bbox_index]
            frame_num2, bbox2 = bbox_list[bbox_index + 1]

            if frame_num1 <= i <= frame_num2:
                curr_iou = calculate_iou(bbox1, bbox2)

                if curr_iou > 0:
                    alpha = (i - frame_num1) / (frame_num2 - frame_num1)
                    interp_bbox = [int(bbox1[j] * (1 - alpha) + bbox2[j] * alpha) for j in range(4)]
                else:
                    interp_bbox = bbox1 if i == frame_num1 else bbox2

                all_bboxes[i] = interp_bbox

                if i == frame_num2:
                    bbox_index += 1
            elif i > frame_num2:
                bbox_index += 1
                if bbox_index >= len(bbox_list) - 1:
                    for k in range(i, total_frames):
                        all_bboxes[k] = bbox_list[-1][1]
                    break
        else:
            all_bboxes[i] = bbox_list[-1][1]

    all_bboxes = [all_bboxes[i] if all_bboxes[i] is not None else all_bboxes[i - 1] for i in range(len(all_bboxes))]
    return all_bboxes


def save_shorts(input_video_path, output_video_path, all_bboxes, original_fps, emoji_mapping, output_width=720, output_height=1280):
    cap = cv2.VideoCapture(input_video_path)
    frame_number = 0

    emoji_counter = 0
    anim_dur = int(1.5*original_fps)
    curr_emoji = ''

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (output_width, output_height))

    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break

        curr_bbox = all_bboxes[frame_number]

        if curr_bbox is not None:
            x1, y1, x2, y2 = curr_bbox

            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2

            desired_aspect = 9 / 16
            frame_height, frame_width = frame_raw.shape[:2]

            crop_h = (y2 - y1) * 2  # Adjust multiplier as needed
            crop_w = crop_h * desired_aspect

            crop_w = min(crop_w, frame_width)
            crop_h = min(crop_h, frame_height)

            crop_x1 = bbox_center_x - crop_w / 2
            crop_y1 = bbox_center_y - crop_h / 2

            crop_x1 = max(0, min(crop_x1, frame_width - crop_w))
            crop_y1 = max(0, min(crop_y1, frame_height - crop_h))
            crop_x2 = crop_x1 + crop_w
            crop_y2 = crop_y1 + crop_h

            crop_x1, crop_y1, crop_x2, crop_y2 = map(int, [crop_x1, crop_y1, crop_x2, crop_y2])
            cropped_frame = frame_raw[crop_y1:crop_y2, crop_x1:crop_x2]
            resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
        else:
            resized_frame = cv2.resize(frame_raw, (output_width, output_height))

        if frame_number in emoji_mapping:
            emoji_counter = 0
            curr_emoji = emoji_mapping[frame_number]
            emoji_y = int((0.5 - 0.1*(emoji_counter / anim_dur))*resized_frame.shape[0])
            draw_emoji(int(0.9*resized_frame.shape[1]), emoji_y, resized_frame, curr_emoji)
        out.write(resized_frame)
        frame_number += 1

    cap.release()
    out.release()

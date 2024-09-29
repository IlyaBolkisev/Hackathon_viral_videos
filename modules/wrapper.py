from modules.audio_processing import get_audio_features
from modules.nlp_features import extract_key_sentences
from modules.transcribation import get_audio_segments
from modules.utils import extract_audio_from_video
from modules.video_processing import get_video_features, extrapolate_bboxes, save_shorts
from modules.emotes_expression import choose_emoji

def get_videos(path_to_video, models):
    path_to_audio = './tmp/audio.mp3'
    extract_audio_from_video(path_to_video, path_to_audio)

    volume_peaks, pitch_changes, music_segments = get_audio_features(path_to_audio)
    extracted_segments = get_audio_segments(path_to_audio)
    all_sentences = extract_key_sentences(extracted_segments)

    activity_frames, emotions_scores, bbox_list, fill_bbox, total_frames, original_fps = \
        get_video_features(path_to_video, models)
    
    emoji_mapping = choose_emoji(emotions_scores)

    all_bboxes = extrapolate_bboxes(bbox_list, total_frames, fill_bbox)
    save_shorts(path_to_video, './tmp/shorts1.mp4', all_bboxes, original_fps, emoji_mapping)

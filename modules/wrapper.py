import os.path

from modules.audio_processing import get_audio_features#, process_music
from modules.nlp_features import extract_key_sentences, get_key_moments
from modules.scoring import VideoSegment, update_nlp_scores, update_scores, rank_segments
from modules.transcribation import get_audio_segments
from modules.utils import extract_audio_from_video, process_activity_scores, process_audio_scores, \
    process_emotions_scores
from modules.video_processing import get_video_features, extrapolate_bboxes, save_shorts


def get_videos(path_to_video, models, music_path=None):
    path_to_audio = './tmp/audio.mp3'
    extract_audio_from_video(path_to_video, path_to_audio)

    volume_peaks, pitch_changes, music_segments = get_audio_features(path_to_audio)
    extracted_segments = get_audio_segments(path_to_audio)
    all_sentences = extract_key_sentences(extracted_segments)
    all_sentences = get_key_moments(extracted_segments, all_sentences)

    activity_frames, emotions_scores, bbox_list, fill_bbox, total_frames, original_fps = \
        get_video_features(path_to_video, models)

    activity_scores = process_activity_scores(activity_frames, total_frames)
    music_score = process_audio_scores(music_segments, original_fps, total_frames)
    pitch_score = process_audio_scores(pitch_changes, original_fps, total_frames)
    volume_score = process_audio_scores(volume_peaks, original_fps, total_frames)
    emotions_scores, emotions_id = process_emotions_scores(emotions_scores, total_frames)

    video_segments = [VideoSegment(s) for s in extracted_segments]
    update_nlp_scores(video_segments, all_sentences)
    update_scores(video_segments, activity_scores, 'activity_score')
    update_scores(video_segments, music_score, 'music_score')
    update_scores(video_segments, pitch_score, 'pitch_score')
    update_scores(video_segments, volume_score, 'volume_score')
    update_scores(video_segments, emotions_scores, 'emotion_score')

    segments = rank_segments(video_segments,
                                    {
                                        'nlp_score': 1.5,
                                        'music_score': 0.25,
                                        'pitch_score': 0.5,
                                        'volume_score': 0.75,
                                        'emotion_score': 1.2,
                                        'activity_score': 1
                                    })

    all_bboxes = extrapolate_bboxes(bbox_list, total_frames, fill_bbox)
    save_shorts(path_to_video, './tmp/shorts1.mp4', all_bboxes, original_fps)

    if os.path.exists('./tmp/music.mp3'):
        music_path = './tmp/music.mp3'
    music_path, video_duration = 'audio/neutral/audio1.mp3', 15
    # process_music(music_path, video_duration)

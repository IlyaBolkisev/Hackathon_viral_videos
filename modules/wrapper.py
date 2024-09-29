import os.path

from moviepy.editor import VideoFileClip, AudioFileClip

from modules.audio_processing import get_audio_features, process_music
from modules.nlp_features import extract_key_sentences, get_key_moments
from modules.scoring import VideoSegment, update_nlp_scores, update_scores, rank_segments
from modules.transcribation import get_audio_segments
from modules.utils import extract_audio_from_video, process_activity_scores, process_audio_scores, \
    process_emotions_scores
from modules.video_processing import get_video_features, extrapolate_bboxes, save_shorts
from modules.emotes_expression import choose_emoji


def get_videos(path_to_video, models, music_path=None):
    path_to_audio = './tmp/audio.mp3'
    extract_audio_from_video(path_to_video, path_to_audio)

    # volume_peaks, pitch_changes, music_segments = get_audio_features(path_to_audio)
    # extracted_segments = get_audio_segments(path_to_audio)
    # all_sentences = extract_key_sentences(extracted_segments)
    # all_sentences = get_key_moments(extracted_segments, all_sentences)

    activity_frames, emotions_scores, bbox_list, fill_bbox, total_frames, original_fps, video_duration = \
        get_video_features(path_to_video, models)
    
    emoji_mapping = choose_emoji(emotions_scores)
    #
    # activity_scores = process_activity_scores(activity_frames, total_frames)
    # music_score = process_audio_scores(music_segments, original_fps, total_frames)
    # pitch_score = process_audio_scores(pitch_changes, original_fps, total_frames)
    # volume_score = process_audio_scores(volume_peaks, original_fps, total_frames)
    # emotions_scores, emotions_id = process_emotions_scores(emotions_scores, total_frames)
    #
    # video_segments = [VideoSegment(s) for s in extracted_segments]
    # update_nlp_scores(video_segments, all_sentences)
    # update_scores(video_segments, activity_scores, 'activity_score')
    # update_scores(video_segments, music_score, 'music_score')
    # update_scores(video_segments, pitch_score, 'pitch_score')
    # update_scores(video_segments, volume_score, 'volume_score')
    # update_scores(video_segments, emotions_scores, 'emotion_score')
    #
    # segments = rank_segments(video_segments,
    #                          {
    #                              'nlp_score': 1.5,
    #                              'music_score': 0.25,
    #                              'pitch_score': 0.5,
    #                              'volume_score': 0.75,
    #                              'emotion_score': 1.2,
    #                              'activity_score': 1
    #                          })

    all_bboxes = extrapolate_bboxes(bbox_list, total_frames, fill_bbox)
    save_shorts(path_to_video, './tmp/shorts1.mp4', all_bboxes, original_fps, emoji_mapping)

    if os.path.exists('./tmp/music.mp3'):
        music_path = './tmp/music.mp3'
    music_path = './audio/neutral/audio1.mp3'

    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, afx

    video = VideoFileClip('./tmp/shorts1.mp4')
    main_audio = AudioFileClip(path_to_audio).set_duration(video.duration)
    bg_audio = AudioFileClip(music_path).fx(afx.audio_loop, duration=video.duration).volumex(0.5)
    combined_audio = CompositeAudioClip([main_audio, bg_audio])
    final_video = video.set_audio(combined_audio)
    final_video.write_videofile(
        './tmp/shorts1_final.mp4',
        codec='libx264',
        audio_codec='aac',
        fps=video.fps
    )



    # combined_path = process_music(path_to_audio, music_path)
    #
    # video_clip = VideoFileClip('./tmp/shorts1.mp4')
    # audio_clip = AudioFileClip(combined_path)
    #
    # min_duration = min(video_clip.duration, audio_clip.duration)
    # trimmed_video = video_clip.subclip(0, min_duration)
    # trimmed_audio = audio_clip.subclip(0, min_duration)
    # final_clip = trimmed_video.set_audio(trimmed_audio)
    # fps_value = 30

    # final_clip = video_clip.set_audio(audio_clip)
    # final_clip.fps = video_clip.fps
    # fps_value = float(final_clip.fps)
    # final_clip.write_videofile('./tmp/shorts1_final.mp4', codec='libx264', audio_codec='aac', fps=fps_value)

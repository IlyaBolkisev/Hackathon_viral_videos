import random
import pysrt
import os.path
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from modules.audio_processing import get_audio_features, process_music
from modules.nlp_features import extract_key_sentences, get_key_moments
from modules.scoring import VideoSegment, update_nlp_scores, update_scores, rank_segments, build_clips
from modules.transcribation import get_audio_segments
from modules.utils import extract_audio_from_video, process_activity_scores, process_audio_scores, \
    process_emotions_scores
from modules.video_processing import get_video_features, extrapolate_bboxes, save_shorts
from modules.emotes_expression import choose_emoji


def get_videos(path_to_video, models, music_path=None):
    path_to_audio = './tmp/audio.mp3'
    extract_audio_from_video(path_to_video, path_to_audio)

    volume_peaks, pitch_changes, music_segments = get_audio_features(path_to_audio)
    extracted_segments = get_audio_segments(path_to_audio)
    all_sentences = extract_key_sentences(extracted_segments)
    all_sentences = get_key_moments(extracted_segments, all_sentences)

    activity_frames, emotions_scores, bbox_list, fill_bbox, total_frames, original_fps, video_duration = \
        get_video_features(path_to_video, models)

    emoji_mapping = choose_emoji(emotions_scores)

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

    builded_clips = build_clips(segments)

    all_bboxes = extrapolate_bboxes(bbox_list, total_frames, fill_bbox)
    save_shorts(path_to_video, './tmp/shorts.mp4', all_bboxes, original_fps, emoji_mapping)

    clips = []
    for clip in builded_clips:
        clips.append([(extracted_segments[idx].start, extracted_segments[idx].end) for idx in sorted(clip)])
    

    def generate_srt(transcript_segments, srt_path):
        subs = []
        for i, segment in enumerate(transcript_segments):
            for word in segment.words:
                start = word.start
                end = word.end
                text = word.word.strip()
                sub = pysrt.SubRipItem(
                    index=i+1,
                    start=pysrt.SubRipTime(seconds=start),
                    end=pysrt.SubRipTime(seconds=end),
                    text=text
                )
                subs.append(sub)
        srt_file = pysrt.SubRipFile(items=subs)
        srt_file.save(srt_path, encoding='utf-8')


    generate_srt(extracted_segments, './tmp/subtitles.srt')
    input_subtitles_path='./tmp/subtitles.srt'
    combine_audio_subtitles_cmd = [
        'ffmpeg',
        '-y',
        '-i', path_to_video,  # Видео без аудио,  # Смешанное аудио
        '-c:v', 'libx264',  # Кодек видео (libx264 рекомендуется для фильтров)
        '-c:a', 'aac',  # Кодек аудио
        '-vf', f"subtitles='{input_subtitles_path}'",  # Добавление субтитров
        '-shortest',  # Обрезать видео по длине аудио, если аудио короче
        './tmp/shorts.mp4'
    ]
    
    subprocess.run(combine_audio_subtitles_cmd, check=True)


    cropped_video = VideoFileClip('./tmp/shorts.mp4')

    for i, clip in enumerate(clips):
        shorts = []
        for start, end in clip:
            if start < 0:
                start = 0
            if end > cropped_video.duration:
                end = cropped_video.duration
            if start < end:
                video = cropped_video.subclip(start, end)
                shorts.append(video)

        final_clip = concatenate_videoclips(shorts)
        
        # Путь к выходному видео
        output_video_path = f"./tmp/output_video{i}.mp4"
        final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

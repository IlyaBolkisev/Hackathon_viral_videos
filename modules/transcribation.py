from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def get_audio_segments():
    model = WhisperModel('large-v3', download_root="./models", device='cuda', compute_type="float16")

    segments, _ = model.transcribe('test25min.wav', language='ru', beam_size=5, word_timestamps=True, vad_filter=True)

    extracted_segments = [s for s in segments]
    
    return extracted_segments
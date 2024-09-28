from faster_whisper import WhisperModel


def get_audio_segments(audio_path):
    model = WhisperModel('large-v3', download_root="./models", device='cuda', compute_type="float16")

    segments, _ = model.transcribe(audio_path, language='ru', beam_size=5, word_timestamps=True, vad_filter=True)
    extracted_segments = [s for s in segments]
    
    return extracted_segments

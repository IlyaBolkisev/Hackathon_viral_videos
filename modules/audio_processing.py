import librosa
import numpy as np


def load_audio_file(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr


def detect_volume_peaks(y, sr, frame_length=2048, hop_length=512, threshold=0.9):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_normalized = rms / np.max(rms)
    peak_indices = np.where(rms_normalized > threshold)[0]
    times = librosa.frames_to_time(peak_indices, sr=sr, hop_length=hop_length)

    return times


def detect_pitch_changes(y, sr, hop_length=512, threshold=0.3):
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    spectral_centroid_normalized = spectral_centroid / np.max(spectral_centroid)
    spectral_delta = np.abs(np.diff(spectral_centroid_normalized))
    change_indices = np.where(spectral_delta > threshold)[0]
    times = librosa.frames_to_time(change_indices, sr=sr, hop_length=hop_length)

    return times


def detect_music_segments(y, sr, hop_length=512, o_env_threshold=0.50):
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    o_env_normalized = o_env / np.max(o_env)
    music_indices = np.where(o_env_normalized > o_env_threshold)[0]
    times = librosa.frames_to_time(music_indices, sr=sr, hop_length=hop_length)

    return times


def get_audio_features(audio_path):
    y, sr = load_audio_file(audio_path)

    volume_peaks = detect_volume_peaks(y, sr)
    pitch_changes = detect_pitch_changes(y, sr)
    music_segments = detect_music_segments(y, sr)

    return volume_peaks, pitch_changes, music_segments


# def process_music(music_path, video_duration):

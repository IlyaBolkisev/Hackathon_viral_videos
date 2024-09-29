import numpy as np


class VideoSegment:
    def __init__(self, segment, fps=30):
        self.segment = segment
        self.start_frame = int(segment.start)*fps
        self.end_frame = int(segment.end)*fps
        self.features = {
            'nlp_score': 0,
            'music_score': 0,
            'pitch_score': 0,
            'volume_score': 0,
            'emotion_score': 0,
            'activity_score': 0,
        }
        self.total_score = 0

    def compute_total_score(self, weights):
        self.total_score = sum([self.features[key]*weights.get(key, 1) for key in self.features])


def update_nlp_scores(segments, nlp_key_moments):
    for moment in nlp_key_moments:
        start = moment['start']
        end = moment['end']
        for segment in segments:
            if segment.start_frame <= end and segment.end_frame >= start:
                segment.features['nlp_score'] += 1


def update_scores(segments, feature_arr, feature_type):
    scored_frames = np.where(feature_arr > 0)
    for frame in scored_frames[0]:
        for segment in segments:
            if segment.start_frame <= frame <= segment.end_frame:
                segment.features[feature_type] += 1

<<<<<<< HEAD
def rank_segments(segments, weights):
    for segment in segments:
        segment.compute_total_score(weights)
    ranked_segments = sorted(segments, lambda x: x.total_score, reverse=True)
    return ranked_segments
=======

def rank_segments(segments, weights):
    for segment in segments:
        segment.compute_total_score(weights)
    return segments
>>>>>>> d96a4c1d7a749e73b1a5bc76bca0b5845293fe22

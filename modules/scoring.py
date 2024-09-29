import numpy as np
from random import randint

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

def rank_segments(segments, weights):
    for segment in segments:
        segment.compute_total_score(weights)
    return segments

def build_clips(extracted_segments):
    def check_segment(idx):
        if 0 <= idx < len(extracted_segments) and ind not in used_segments:
            clip_buffer.append(idx)
            used_segments.append(idx)
        curr_len = sum([extracted_segments[i].end - extracted_segments[i].start for i in clip_buffer])

        return True if curr_len < vid_length else False
        
    max_score = max([s.total_score for s in extracted_segments])
    normalized_scores = [s.total_score / max_score for s in extracted_segments]
    sorted_indices = sorted(range(len(extracted_segments)), key=lambda i: normalized_scores[i], reverse=True)
    
    full_len = sum([s.end - s.start for s in extracted_segments])
    vid_length = min(randint(10, 180), 0.5*full_len)
    summary_len = 0
    clips_pool = []
    used_segments = []
    clip_buffer = []
    for ind in sorted_indices:
        if len(clips_pool) >= 20 or summary_len >= 0.5*full_len:
            break
        flag = check_segment(ind)
        flag = check_segment(ind-1)
        flag = check_segment(ind+1)
        for i in [ind, ind-1, ind+1]:
            flag = check_segment(i)
            if not flag:
                clips_pool.append(clip_buffer.copy())
                summary_len += sum(extracted_segments[j].end - extracted_segments[j].start for j in clip_buffer)
                clip_buffer = []
                break
    return clips_pool
        
        

# def build_clips(segments, vid_length):
#     max_score = max([s.total_score for s in segments])
#     normalized_scores = [s.total_score / max_score for s in segments]
#     sorted_indices = sorted(range(len(segments)), key=lambda i: normalized_scores[i], reverse=True)
     
#     clips_pool = []
#     used_segments = []
#     clip_buffer = []
#     duration_time_thresh = min(randint(10, 180), 0.5*vid_length)
#     for ind in sorted_indices:
#         if len(clip_buffer) == 0:
#             clip_buffer.append(ind)
#             used_segments.append(ind)
#             continue
#         score_neighbors

# def score_neighbors(segments, clip_buffer, normalized_scores):

#     for ind in clip_buffer:
#         if ind == 0:
#             neighbor_sum = normalized_scores[ind] + normalized_scores[ind+1]
#         elif ind == len(normalized_scores)-1:
#             neighbor_sum = normalized_scores[ind] + normalized_scores[ind-1]
#         else:
#             neighbor_sum = normalized_scores[ind] + normalized_scores[ind-1] + normalized_scores[ind+1]
        

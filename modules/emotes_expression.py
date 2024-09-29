import numpy as np
from PIL import Image


def choose_emoji(emotion_scores):
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emoji_mapping = {}
    for frame_i, emotion_vector in emotion_scores:
        argmax_emote = np.argmax(emotion_vector)
        if emotion_vector.max() > 0.95 and argmax_emote != len(labels) - 1:
            emoji_mapping[frame_i] = labels[argmax_emote]
    
    return emoji_mapping


def draw_emoji(x, y, frame, em_type):
    background = Image.fromarray(frame)

    emoji_im = Image.open(f'./emotes/{em_type}.png').resize((70, 70))
    emoji_im = emoji_im.convert('RGBA')
    
    background.paste(emoji_im, (x, y), mask=emoji_im)

    return np.array(background)

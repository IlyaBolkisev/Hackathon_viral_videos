import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('punkt_tab')


def extract_key_sentences(transcript_segments):  # SentenceTransformer variant
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 'all-MiniLM-L6-v2'
    sentences = [seg.text.strip() for seg in transcript_segments]
    embeddings = model.encode(sentences)
    mean_emb = np.mean(embeddings, axis=0)
    dists = [np.linalg.norm(emb - mean_emb) for emb in embeddings]
    dist_mean = np.mean(dists)
    dist_std = np.std(dists) * 0.2
    dist_deviations = [d - dist_mean for d in dists]

    return [sent.text.strip() for i, sent in enumerate(transcript_segments) if np.abs(dist_deviations[i]) < dist_std]

from transformers import pipeline
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def extract_key_sentences(transcript_segments): # HuggingFace pipeline
    summarizer = pipeline('summarization', device='cuda', model='IlyaGusev/rut5_base_headline_gen_telegram')
    tokenizer = summarizer.tokenizer
    all_sentences = []
    tokens_buffer_len = 0
    text = ''
    for segment in transcript_segments:
        seg_length = len(tokenizer(segment.text).input_ids)
        if tokens_buffer_len + seg_length >= summarizer.model.config.max_length:
            summary = summarizer(text, do_sample=False)[0]['summary_text']
            sents = nltk.sent_tokenize(text, language='russian')
            key_sents = [sent for sent in sents if sent in summary]
            all_sentences.extend(key_sents)
            text = ''
            tokens_buffer_len = 0
        
        text = text + segment.text
        tokens_buffer_len += seg_length

    return all_sentences


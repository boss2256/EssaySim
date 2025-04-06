# utils/similarity.py
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load the transformer model only
model = SentenceTransformer("./sentence-transformers/all-MiniLM-L6-v2")
print("✅ Loaded transformer model: sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text, remove_stopwords=False):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]
    return words

def encode_sentence(text):
    return model.encode(text, convert_to_tensor=True)

def calculate_similarity(text1, text2):
    emb1 = encode_sentence(text1)
    emb2 = encode_sentence(text2)
    score = util.cos_sim(emb1, emb2).item()
    print("✅ Transformer similarity used")
    return score

def word_level_contribution(text1, text2, remove_stopwords=False):
    words1 = clean_text(text1, remove_stopwords)
    words2 = clean_text(text2, remove_stopwords)

    max_words = 30
    if len(words1) > max_words:
        words1 = words1[:max_words // 3] + \
                 words1[len(words1)//2 - max_words//6:len(words1)//2 + max_words//6] + \
                 words1[-max_words // 3:]
    if len(words2) > max_words:
        words2 = words2[:max_words // 3] + \
                 words2[len(words2)//2 - max_words//6:len(words2)//2 + max_words//6] + \
                 words2[-max_words // 3:]

    words1 = [w for w in words1 if len(w) >= 3]
    words2 = [w for w in words2 if len(w) >= 3]

    embeddings1 = model.encode(words1, convert_to_tensor=True)
    embeddings2 = model.encode(words2, convert_to_tensor=True)

    sim_matrix = util.cos_sim(embeddings1, embeddings2)
    contrib_scores = []

    for i, w1 in enumerate(words1):
        best_idx = sim_matrix[i].argmax().item()
        sim_score = sim_matrix[i][best_idx].item()
        if sim_score > 0.3:
            contrib_scores.append((w1, words2[best_idx], sim_score))

    contrib_scores = sorted(contrib_scores, key=lambda x: x[2], reverse=True)
    return contrib_scores

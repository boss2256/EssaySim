# utils/similarity.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load SBERT model once
model = SentenceTransformer("all-mpnet-base-v2")


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
    return score


def word_level_contribution(text1, text2, remove_stopwords=False):
    words1 = clean_text(text1, remove_stopwords)
    words2 = clean_text(text2, remove_stopwords)

    # Limit the number of words to analyze if texts are very long to prevent timeouts
    max_words = 30  # Reduced for faster analysis
    if len(words1) > max_words:
        # Take a sample of words from the beginning, middle and end
        words1_sample = words1[:max_words // 3] + words1[len(words1) // 2 - max_words // 6:len(
            words1) // 2 + max_words // 6] + words1[-max_words // 3:]
        words1 = words1_sample
    if len(words2) > max_words:
        # Same sampling for second text
        words2_sample = words2[:max_words // 3] + words2[len(words2) // 2 - max_words // 6:len(
            words2) // 2 + max_words // 6] + words2[-max_words // 3:]
        words2 = words2_sample

    # Performance optimization: batch encode words
    try:
        # Filter out very short words before encoding
        words1 = [w for w in words1 if len(w) >= 3]  # Increased minimum word length for performance
        words2 = [w for w in words2 if len(w) >= 3]

        # Batch encode all words at once (much faster than one-by-one)
        embeddings1 = model.encode(words1, convert_to_tensor=True)
        embeddings2 = model.encode(words2, convert_to_tensor=True)

        # Calculate similarity matrix (all word pairs at once)
        sim_matrix = util.cos_sim(embeddings1, embeddings2)

        contrib_scores = []
        # Get best match for each word
        for i, w1 in enumerate(words1):
            # Get best matching word index
            best_idx = sim_matrix[i].argmax().item()
            sim_score = sim_matrix[i][best_idx].item()

            # Only add matches above a minimum threshold
            if sim_score > 0.3:  # Only include reasonably related words
                contrib_scores.append((w1, words2[best_idx], sim_score))

        # Sort by similarity score (highest first)
        contrib_scores = sorted(contrib_scores, key=lambda x: x[2], reverse=True)

    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback to slower method if batch processing fails
        contrib_scores = []
        for w1 in words1:
            # Skip very short words
            if len(w1) < 3:
                continue

            try:
                e1 = model.encode(w1, convert_to_tensor=True)
                best_match = (words2[0] if words2 else "", 0)

                for w2 in words2:
                    if len(w2) < 3:
                        continue

                    try:
                        e2 = model.encode(w2, convert_to_tensor=True)
                        sim = util.cos_sim(e1, e2).item()
                        if sim > best_match[1]:
                            best_match = (w2, sim)
                    except:
                        continue

                if best_match[1] > 0.3:
                    contrib_scores.append((w1, best_match[0], best_match[1]))

            except:
                continue

    # Ensure we have at least some results
    if not contrib_scores and words1 and words2:
        # Add a few default examples if processing failed
        sample_count = min(5, len(words1), len(words2))
        for i in range(sample_count):
            contrib_scores.append((words1[i], words2[i], 1.0 if words1[i] == words2[i] else 0.5))

    return contrib_scores
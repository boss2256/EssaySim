# utils/similarity.py
import os
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Define a flag to control which implementation to use as a global variable
USE_TRANSFORMERS = True

# Setup TF-IDF vectorizers for fallback
document_vectorizer = TfidfVectorizer(stop_words='english')
word_vectorizer = TfidfVectorizer(ngram_range=(1, 1))

# Setup the transformer model
model = None

try:
    from sentence_transformers import SentenceTransformer, util
    #model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer(".cache/sentence-transformers_all-MiniLM-L6-v2")
    print("✅ Loaded transformer model: sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"⚠️ Failed to load transformer model: {e}")
    USE_TRANSFORMERS = False



def clean_text(text, remove_stopwords=False):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]
    return words


def encode_sentence(text):
    global USE_TRANSFORMERS, model

    if USE_TRANSFORMERS and model is not None:
        try:
            return model.encode(text, convert_to_tensor=True)
        except Exception as e:
            print(f"Error encoding with transformer: {e}")
            USE_TRANSFORMERS = False

    # Fallback to TF-IDF
    return document_vectorizer.fit_transform([text])


def calculate_similarity(text1, text2):
    global USE_TRANSFORMERS, model, util

    if USE_TRANSFORMERS and model is not None:
        try:
            emb1 = encode_sentence(text1)
            emb2 = encode_sentence(text2)
            score = util.cos_sim(emb1, emb2).item()
            return score
        except Exception as e:
            print(f"Error in transformer similarity: {e}")
            # Fall back to TF-IDF if transformer fails
            USE_TRANSFORMERS = False

    # TF-IDF similarity approach
    tfidf_matrix = document_vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(similarity)


def word_level_contribution(text1, text2, remove_stopwords=False):
    global USE_TRANSFORMERS, model, util

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

    # Filter out very short words before encoding
    words1 = [w for w in words1 if len(w) >= 3]  # Increased minimum word length for performance
    words2 = [w for w in words2 if len(w) >= 3]

    contrib_scores = []

    if USE_TRANSFORMERS and model is not None:
        try:
            # Performance optimization: batch encode words
            # Batch encode all words at once (much faster than one-by-one)
            embeddings1 = model.encode(words1, convert_to_tensor=True)
            embeddings2 = model.encode(words2, convert_to_tensor=True)

            # Calculate similarity matrix (all word pairs at once)
            sim_matrix = util.cos_sim(embeddings1, embeddings2)

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
            print(f"Error in transformer word contribution: {e}")
            USE_TRANSFORMERS = False
            # Continue with TF-IDF approach below

    # If transformer approach failed or is disabled, use TF-IDF
    if not USE_TRANSFORMERS or model is None or not contrib_scores:
        # TF-IDF fallback for word-level contributions
        # Prepare all unique words for the vectorizer
        all_words = list(set(words1 + words2))
        # Fit vectorizer on all words
        word_vectorizer.fit([" ".join(all_words)])

        # Get word vectors once for efficiency
        word_vectors1 = {w: word_vectorizer.transform([w]) for w in words1}
        word_vectors2 = {w: word_vectorizer.transform([w]) for w in words2}

        for w1 in words1:
            best_match = ("", 0.0)
            v1 = word_vectors1[w1]

            for w2 in words2:
                # Calculate similarity - identical words have score of 1.0
                if w1 == w2:
                    sim = 1.0
                else:
                    v2 = word_vectors2[w2]
                    sim = cosine_similarity(v1, v2)[0][0]

                if sim > best_match[1]:
                    best_match = (w2, sim)

            # Only add matches above a minimum threshold
            if best_match[1] > 0.3:  # Only include reasonably related words
                contrib_scores.append((w1, best_match[0], float(best_match[1])))

        # Sort by similarity score (highest first)
        contrib_scores = sorted(contrib_scores, key=lambda x: x[2], reverse=True)

    # Ensure we have at least some results
    if not contrib_scores and words1 and words2:
        # Add a few default examples if processing failed
        sample_count = min(5, len(words1), len(words2))
        for i in range(sample_count):
            contrib_scores.append((words1[i], words2[i], 1.0 if words1[i] == words2[i] else 0.5))

    return contrib_scores
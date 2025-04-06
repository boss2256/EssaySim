from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("sentence-transformers/all-MiniLM-L6-v2")  # save in proper format

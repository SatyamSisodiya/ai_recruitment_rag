# config.py (updated)
import os

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Default to a broadly available model on v1beta; can be overridden via env
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")
GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.0"))

# Embeddings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Paths
INDEX_PATH = "data/faiss_index.bin"
METADATA_DB = "data/metadata.parquet"
RESUME_DIR = "data/resumes"
JD_DIR = "data/jds"

# Chunking / retrieval
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 5

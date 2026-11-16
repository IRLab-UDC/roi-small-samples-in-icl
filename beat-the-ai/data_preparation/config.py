"""
Configuration file for SIGIR ICL experiments.

This file contains all the hyperparameters and settings for the 
In-Context Learning experiments with semantic similarity.
"""

# Elasticsearch Configuration
ES_HOST = "10.56.35.149"
ES_PORT = 9208
ES_INDEX_NAME = "anxo_adversarial_qa_jina_training_samples"
ES_SCHEME = "http"  # Change to "https" if using SSL

# Sentence Transformer Configuration
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"  # 1024 dimensions
EMBEDDING_DIM = 1024
BATCH_SIZE = 32  # For encoding

# Index Mapping Configuration
INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "30s"
}

# Data paths
BEAT_THE_AI_TRAIN_PATH = "/mnt/datasets/qa/aqa_v1.0/combined/train.json" # ~130,000+ examples in total (Answerable Questions: ~86,000+ (from v1.1))


# # Experiment Configuration
# MAX_CONTEXT_LENGTH = 512  # Max tokens for context
# TOP_K_SIMILAR = [1, 3, 5, 10]  # Number of similar examples to retrieve
# SIMILARITY_METRICS = ["query", "context", "query_context"]  # Which embedding to use

# # Random seed for reproducibility
# RANDOM_SEED = 42

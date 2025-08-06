from enum import Enum


class ChunkingStrategy(str, Enum):
    PAGE = "page_chunking"
    TOKEN = "token_chunking"
    SEMANTIC = "semantic_chunking"


class QueryEnhancement(str, Enum):
    HYDE = "hyde"
    STEP_BACK = "step_back_prompting"
    MULTI_QUERY = "multi_query"


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MAX_SIM = "max_sim"


class RoutingStrategy(str, Enum):
    SEMANTIC = "semantic"

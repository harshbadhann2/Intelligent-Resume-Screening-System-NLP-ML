"""
Resume Screening System - NLP & ML based resume ranking
"""

__version__ = "0.1.0"
__author__ = "Data Science Team"

# Import main components
try:
    from .preprocessor import TextPreprocessor
    from .embeddings import TFIDFEmbedder, Word2VecEmbedder, BERTEmbedder
    from .similarity import SimilarityScorer
    from .ranker import RankingModel
    from .data_loader import DataLoader
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

__all__ = [
    'TextPreprocessor',
    'TFIDFEmbedder',
    'Word2VecEmbedder', 
    'BERTEmbedder',
    'SimilarityScorer',
    'RankingModel',
    'DataLoader',
]

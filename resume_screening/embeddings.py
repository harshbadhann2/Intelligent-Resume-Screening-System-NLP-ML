"""
Embedding Generation Module - TF-IDF, Word2Vec, and BERT
"""

import numpy as np
from typing import List, Union, Tuple
import pickle
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

# Optional imports for deep learning embeddings
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None

try:
    import torch
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    torch = None
    SentenceTransformer = None

from .preprocessor import TextPreprocessor


class BaseEmbedder:
    """Base class for embedding models"""
    
    def __init__(self, name: str):
        self.name = name
        self.preprocessor = TextPreprocessor()
        self.model = None
        
    def train(self, documents: List[str]):
        """Train embedding model"""
        raise NotImplementedError
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for text"""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model"""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load model"""
        raise NotImplementedError


class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF vectorizer for text embedding"""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        """
        Initialize TF-IDF embedder
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (min, max)
        """
        super().__init__("TF-IDF")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.model = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.is_trained = False
    
    def train(self, documents: List[str]):
        """Train TF-IDF model on documents"""
        # Preprocess documents
        processed_docs = [' '.join(self.preprocessor.process(doc)) 
                         for doc in documents]
        
        # Fit model
        self.model.fit(processed_docs)
        self.is_trained = True
        
        print(f"TF-IDF model trained on {len(documents)} documents")
        print(f"Vocabulary size: {len(self.model.vocabulary_)}")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get TF-IDF embeddings
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Sparse matrix or array of embeddings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before embedding")
        
        if isinstance(text, str):
            text = [text]
        
        # Preprocess
        processed = [' '.join(self.preprocessor.process(t)) for t in text]
        
        # Transform
        embeddings = self.model.transform(processed)
        
        return embeddings.toarray() if len(text) > 1 else embeddings.toarray()[0]
    
    def save(self, path: str):
        """Save TF-IDF model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(path, 'tfidf_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"TF-IDF model saved to {model_path}")
    
    def load(self, path: str):
        """Load TF-IDF model"""
        model_path = os.path.join(path, 'tfidf_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"TF-IDF model loaded from {model_path}")


class Word2VecEmbedder(BaseEmbedder):
    """Word2Vec embedder using Gensim"""
    
    def __init__(self, vector_size: int = 300, window: int = 5, 
                 min_count: int = 2, sg: int = 1):
        """
        Initialize Word2Vec embedder
        
        Args:
            vector_size: Dimension of word vectors
            window: Context window size
            min_count: Minimum word frequency
            sg: 0=CBOW, 1=Skip-gram
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for Word2VecEmbedder. Install with: pip install gensim")
        super().__init__("Word2Vec")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None
        self.is_trained = False
    
    def train(self, documents: List[str], epochs: int = 5):
        """
        Train Word2Vec model
        
        Args:
            documents: List of text documents
            epochs: Training epochs
        """
        # Preprocess documents into sentences
        sentences = []
        for doc in documents:
            tokens = self.preprocessor.process(doc)
            sentences.append(tokens)
        
        # Train model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=epochs,
            workers=4
        )
        self.is_trained = True
        
        print(f"Word2Vec model trained on {len(sentences)} documents")
        print(f"Vocabulary size: {len(self.model.wv)}")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get Word2Vec embeddings (document-level via averaging)
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Document embedding(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before embedding")
        
        if isinstance(text, str):
            text = [text]
            single = True
        else:
            single = False
        
        embeddings = []
        for t in text:
            tokens = self.preprocessor.process(t)
            
            # Get word vectors and average
            valid_vectors = [self.model.wv[token] for token in tokens 
                            if token in self.model.wv]
            
            if valid_vectors:
                doc_embedding = np.mean(valid_vectors, axis=0)
            else:
                doc_embedding = np.zeros(self.vector_size)
            
            embeddings.append(doc_embedding)
        
        embeddings = np.array(embeddings)
        return embeddings[0] if single else embeddings
    
    def save(self, path: str):
        """Save Word2Vec model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(path, 'word2vec_model.bin')
        self.model.save(model_path)
        print(f"Word2Vec model saved to {model_path}")
    
    def load(self, path: str):
        """Load Word2Vec model"""
        model_path = os.path.join(path, 'word2vec_model.bin')
        self.model = Word2Vec.load(model_path)
        self.is_trained = True
        print(f"Word2Vec model loaded from {model_path}")


class BERTEmbedder(BaseEmbedder):
    """BERT embedder using Sentence Transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize BERT embedder
        
        Args:
            model_name: Hugging Face model name
        """
        if not BERT_AVAILABLE:
            raise ImportError("sentence-transformers and torch are required for BERTEmbedder. "
                            "Install with: pip install sentence-transformers torch")
        super().__init__("BERT")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.is_trained = True  # Pre-trained model
        
        print(f"BERT model loaded: {model_name}")
    
    def train(self, documents: List[str]):
        """
        BERT is pre-trained, but this method can fine-tune if needed
        Currently just passes through
        """
        print(f"BERT is pre-trained. Fine-tuning not implemented in this version.")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get BERT embeddings using Sentence Transformers
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Document embedding(s)
        """
        if isinstance(text, str):
            text = [text]
            single = True
        else:
            single = False
        
        # Preprocess text
        processed_text = [' '.join(self.preprocessor.process(t)) for t in text]
        
        # Get embeddings
        embeddings = self.model.encode(processed_text, convert_to_numpy=True)
        
        return embeddings[0] if single else embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        emb1 = self.embed(text1).reshape(1, -1)
        emb2 = self.embed(text2).reshape(1, -1)
        
        sim = cosine_similarity(emb1, emb2)[0][0]
        return float(sim)
    
    def save(self, path: str):
        """Save BERT model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"BERT model saved to {path}")
    
    def load(self, path: str):
        """Load BERT model"""
        self.model = SentenceTransformer(path)
        self.is_trained = True
        print(f"BERT model loaded from {path}")


class EmbeddingFactory:
    """Factory for creating embedding models"""
    
    @staticmethod
    def create(model_type: str, **kwargs) -> BaseEmbedder:
        """
        Create embedding model
        
        Args:
            model_type: Type of model ('tfidf', 'word2vec', 'bert')
            **kwargs: Model-specific arguments
            
        Returns:
            Initialized embedder
        """
        if model_type.lower() == 'tfidf':
            return TFIDFEmbedder(**kwargs)
        elif model_type.lower() == 'word2vec':
            return Word2VecEmbedder(**kwargs)
        elif model_type.lower() == 'bert':
            return BERTEmbedder(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

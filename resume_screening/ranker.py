"""
Ranking and Classification Module - Resume ranking models
"""

import numpy as np
from typing import List, Tuple, Union
import pickle
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline

from .embeddings import BaseEmbedder, EmbeddingFactory
from .similarity import SimilarityScorer


class RankingModel:
    """Base ranking model for resume scoring"""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize ranking model
        
        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying model"""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the ranking model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"{self.model_type} model trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability scores
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        return metrics
    
    def save(self, path: str):
        """Save model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(path, f'{self.model_type}_model.pkl')
        scaler_path = os.path.join(path, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        model_path = os.path.join(path, f'{self.model_type}_model.pkl')
        scaler_path = os.path.join(path, 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {path}")


class NeuralNetworkRanker(nn.Module):
    """Neural network for ranking resumes"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None):
        """
        Initialize neural network ranker
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
        """
        super(NeuralNetworkRanker, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)


class ResumeRanker:
    """
    Complete resume ranking system combining embeddings and classification
    """
    
    def __init__(self, embedder_type: str = 'bert', 
                 model_type: str = 'gradient_boosting'):
        """
        Initialize resume ranker
        
        Args:
            embedder_type: Type of embedder to use
            model_type: Type of ranking model to use
        """
        self.embedder = EmbeddingFactory.create(embedder_type)
        self.scorer = SimilarityScorer(self.embedder)
        self.ranker = RankingModel(model_type)
        self.embedder_type = embedder_type
        self.model_type = model_type
    
    def extract_features(self, resume_text: str, 
                        job_description: str) -> np.ndarray:
        """
        Extract features for ranking model
        
        Args:
            resume_text: Resume text
            job_description: Job description
            
        Returns:
            Feature vector
        """
        # Get embeddings
        resume_emb = self.embedder.embed(resume_text)
        job_emb = self.embedder.embed(job_description)
        
        # Compute similarity metrics
        cosine_sim = self.scorer.cosine_similarity(resume_emb, job_emb)
        euclidean_sim = self.scorer.euclidean_similarity(resume_emb, job_emb)
        
        # Concatenate embeddings and similarities
        features = np.concatenate([
            resume_emb.flatten(),
            job_emb.flatten(),
            [cosine_sim, euclidean_sim]
        ])
        
        return features
    
    def train(self, resume_texts: List[str], job_descriptions: List[str],
             labels: np.ndarray):
        """
        Train ranking model
        
        Args:
            resume_texts: List of resume texts
            job_descriptions: List of corresponding job descriptions
            labels: Binary labels (1=good match, 0=bad match)
        """
        # Extract features
        X = np.array([self.extract_features(resume, job) 
                     for resume, job in zip(resume_texts, job_descriptions)])
        
        # Train model
        self.ranker.train(X, labels)
    
    def rank_resumes(self, resume_texts: List[str], 
                    job_description: str) -> List[Tuple[int, float]]:
        """
        Rank resumes against job description
        
        Args:
            resume_texts: List of resume texts
            job_description: Job description
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        scores = []
        
        for idx, resume_text in enumerate(resume_texts):
            features = self.extract_features(resume_text, job_description)
            features = features.reshape(1, -1)
            
            # Get prediction probability
            proba = self.ranker.predict_proba(features)
            score = float(proba[0, 1])  # Probability of positive class
            
            scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def save(self, path: str):
        """Save ranker"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save embedder
        embedder_path = os.path.join(path, 'embedder')
        self.embedder.save(embedder_path)
        
        # Save ranker
        self.ranker.save(path)
        
        # Save metadata
        metadata = {
            'embedder_type': self.embedder_type,
            'model_type': self.model_type
        }
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Resume ranker saved to {path}")
    
    def load(self, path: str):
        """Load ranker"""
        # Load metadata
        with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Recreate ranker with same configuration
        self.embedder_type = metadata['embedder_type']
        self.model_type = metadata['model_type']
        
        # Load embedder
        embedder_path = os.path.join(path, 'embedder')
        self.embedder = EmbeddingFactory.create(self.embedder_type)
        self.embedder.load(embedder_path)
        
        # Load ranker
        self.ranker = RankingModel(self.model_type)
        self.ranker.load(path)
        
        # Recreate scorer
        self.scorer = SimilarityScorer(self.embedder)
        
        print(f"Resume ranker loaded from {path}")

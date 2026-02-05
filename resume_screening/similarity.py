"""
Similarity Scoring Module - Multiple similarity metrics
"""

import numpy as np
from typing import Union, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .embeddings import BaseEmbedder, BERTEmbedder, TFIDFEmbedder, Word2VecEmbedder


class SimilarityScorer:
    """
    Compute similarity between resume and job description using multiple metrics
    """
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize similarity scorer
        
        Args:
            embedder: Embedding model to use
        """
        self.embedder = embedder
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Ensure 2D arrays
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def euclidean_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance and convert to similarity
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Euclidean similarity score (1/(1+distance))
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        distance = euclidean_distances(emb1, emb2)[0][0]
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    def dot_product_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute dot product similarity
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Dot product similarity
        """
        if emb1.ndim > 1:
            emb1 = emb1[0]
        if emb2.ndim > 1:
            emb2 = emb2[0]
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def score_resume(self, resume_text: str, job_description: str,
                    metric: str = 'cosine') -> float:
        """
        Score resume against job description
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        resume_emb = self.embedder.embed(resume_text)
        job_emb = self.embedder.embed(job_description)
        
        # Compute similarity
        if metric.lower() == 'cosine':
            score = self.cosine_similarity(resume_emb, job_emb)
        elif metric.lower() == 'euclidean':
            score = self.euclidean_similarity(resume_emb, job_emb)
        elif metric.lower() == 'dot':
            score = self.dot_product_similarity(resume_emb, job_emb)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Clamp to [0, 1]
        score = max(0, min(1, score))
        return score
    
    def score_multiple_resumes(self, resume_texts: List[str], 
                               job_description: str,
                               metric: str = 'cosine') -> List[Tuple[int, float]]:
        """
        Score multiple resumes against job description
        
        Args:
            resume_texts: List of resume texts
            job_description: Job description text
            metric: Similarity metric
            
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        scores = []
        job_emb = self.embedder.embed(job_description)
        
        for idx, resume_text in enumerate(resume_texts):
            resume_emb = self.embedder.embed(resume_text)
            
            if metric.lower() == 'cosine':
                score = self.cosine_similarity(resume_emb, job_emb)
            elif metric.lower() == 'euclidean':
                score = self.euclidean_similarity(resume_emb, job_emb)
            elif metric.lower() == 'dot':
                score = self.dot_product_similarity(resume_emb, job_emb)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            score = max(0, min(1, score))
            scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores


class MultiMetricScorer:
    """
    Combine multiple similarity metrics for robust scoring
    """
    
    def __init__(self, embedders: dict):
        """
        Initialize multi-metric scorer
        
        Args:
            embedders: Dictionary of embedders {name: embedder}
        """
        self.embedders = embedders
        self.scorers = {name: SimilarityScorer(emb) 
                       for name, emb in embedders.items()}
        self.weights = {name: 1.0 / len(embedders) 
                       for name in embedders.keys()}
    
    def set_weights(self, weights: dict):
        """
        Set weights for different embedders
        
        Args:
            weights: Dictionary of weights {embedder_name: weight}
        """
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
    
    def score_resume(self, resume_text: str, job_description: str) -> dict:
        """
        Score resume using multiple metrics
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with scores from each embedder and combined score
        """
        scores = {}
        
        for name, scorer in self.scorers.items():
            score = scorer.score_resume(resume_text, job_description, 
                                       metric='cosine')
            scores[name] = score
        
        # Compute weighted average
        combined_score = sum(scores[name] * self.weights[name] 
                           for name in scores.keys())
        
        scores['combined'] = combined_score
        
        return scores
    
    def score_multiple_resumes(self, resume_texts: List[str],
                              job_description: str) -> List[dict]:
        """
        Score multiple resumes using multiple metrics
        
        Args:
            resume_texts: List of resume texts
            job_description: Job description text
            
        Returns:
            List of dictionaries with scores for each resume
        """
        all_scores = []
        
        for idx, resume_text in enumerate(resume_texts):
            scores = self.score_resume(resume_text, job_description)
            scores['index'] = idx
            all_scores.append(scores)
        
        # Sort by combined score
        all_scores.sort(key=lambda x: x['combined'], reverse=True)
        
        return all_scores


class SemanticSimilarityScorer:
    """
    Advanced semantic similarity using BERT embeddings
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic scorer with BERT
        
        Args:
            model_name: Hugging Face model name
        """
        self.embedder = BERTEmbedder(model_name)
        self.scorer = SimilarityScorer(self.embedder)
    
    def score_resume(self, resume_text: str, job_description: str) -> float:
        """
        Score resume using semantic similarity
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Semantic similarity score
        """
        return self.scorer.score_resume(resume_text, job_description, 
                                       metric='cosine')
    
    def score_section_matching(self, resume_text: str, 
                              job_description: str) -> dict:
        """
        Score matching between resume sections and job requirements
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with section-wise scores
        """
        # Split into sections
        resume_sections = resume_text.split('\n\n')
        job_sections = job_description.split('\n\n')
        
        section_scores = []
        
        for resume_section in resume_sections:
            section_scores_list = []
            for job_section in job_sections:
                score = self.scorer.score_resume(resume_section, job_section)
                section_scores_list.append(score)
            
            # Best match for this resume section
            if section_scores_list:
                best_score = max(section_scores_list)
                section_scores.append(best_score)
        
        # Overall score is average of section best matches
        overall_score = np.mean(section_scores) if section_scores else 0.0
        
        return {
            'overall_score': overall_score,
            'section_scores': section_scores,
            'avg_section_score': overall_score
        }

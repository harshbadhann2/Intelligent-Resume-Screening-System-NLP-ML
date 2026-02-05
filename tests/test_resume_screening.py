"""
Unit tests for resume screening system
"""

import pytest
import numpy as np
from resume_screening.preprocessor import TextPreprocessor, ResumeParser
from resume_screening.embeddings import TFIDFEmbedder, Word2VecEmbedder, BERTEmbedder
from resume_screening.similarity import SimilarityScorer
from resume_screening.ranker import RankingModel


class TestPreprocessor:
    """Test TextPreprocessor class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "Hello, this is a TEST! Visit https://example.com or email me@example.com"
        cleaned = self.preprocessor.clean_text(text)
        
        assert "https" not in cleaned
        assert "example.com" not in cleaned
        assert "hello" in cleaned  # Should be lowercase
        assert len(cleaned) > 0
    
    def test_tokenize(self):
        """Test tokenization"""
        text = "This is a sample text for testing"
        tokens = self.preprocessor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_process(self):
        """Test complete preprocessing"""
        text = "This is a SAMPLE text with STOPWORDS that should be removed"
        tokens = self.preprocessor.process(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_extract_skills(self):
        """Test skill extraction"""
        text = "Expert in Python and JavaScript. Strong with Django and React."
        skills = TextPreprocessor.extract_skills(text)
        
        assert len(skills) > 0
        assert 'python' in [s.lower() for s in skills]
    
    def test_extract_emails(self):
        """Test email extraction"""
        text = "Contact me at john@example.com or jane@test.org"
        emails = TextPreprocessor.extract_emails(text)
        
        assert len(emails) == 2
        assert 'john@example.com' in emails
    
    def test_extract_phones(self):
        """Test phone number extraction"""
        text = "Call me at (555) 123-4567 or 555.987.6543"
        phones = TextPreprocessor.extract_phones(text)
        
        assert len(phones) >= 1


class TestEmbedders:
    """Test embedding models"""
    
    def test_tfidf_training(self):
        """Test TF-IDF training"""
        documents = [
            "machine learning is important",
            "deep learning networks",
            "natural language processing"
        ]
        
        tfidf = TFIDFEmbedder()
        tfidf.train(documents)
        
        assert tfidf.is_trained
    
    def test_tfidf_embedding(self):
        """Test TF-IDF embedding"""
        documents = ["test document one", "test document two", "another test"]
        
        tfidf = TFIDFEmbedder()
        tfidf.train(documents)
        
        emb = tfidf.embed("test document")
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0
    
    def test_word2vec_training(self):
        """Test Word2Vec training"""
        documents = [
            "machine learning is awesome",
            "deep learning with networks",
            "natural language processing"
        ]
        
        w2v = Word2VecEmbedder()
        w2v.train(documents, epochs=5)
        
        assert w2v.is_trained
        assert w2v.model is not None
    
    def test_word2vec_embedding(self):
        """Test Word2Vec embedding"""
        documents = ["test document one", "test document two", "another test"]
        
        w2v = Word2VecEmbedder(vector_size=100)
        w2v.train(documents)
        
        emb = w2v.embed("test document")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (100,)
    
    def test_bert_embedding(self):
        """Test BERT embedding"""
        bert = BERTEmbedder()
        
        emb = bert.embed("This is a test sentence")
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0
    
    def test_bert_similarity(self):
        """Test BERT similarity"""
        bert = BERTEmbedder()
        
        sim = bert.similarity("I am a software engineer", 
                            "I am a software developer")
        assert 0 <= sim <= 1
        assert sim > 0.5  # Should be similar


class TestSimilarityScorer:
    """Test similarity scoring"""
    
    def test_cosine_similarity(self):
        """Test cosine similarity"""
        embedder = TFIDFEmbedder()
        embedder.train(["test one", "test two", "test three"])
        scorer = SimilarityScorer(embedder)
        
        emb1 = np.array([1, 0, 0])
        emb2 = np.array([1, 0, 0])
        
        sim = scorer.cosine_similarity(emb1, emb2)
        assert sim == pytest.approx(1.0)
    
    def test_score_multiple_resumes(self):
        """Test scoring multiple resumes"""
        embedder = BERTEmbedder()
        scorer = SimilarityScorer(embedder)
        
        resumes = ["Python developer", "Java developer", "Data scientist"]
        job = "Looking for Python expert"
        
        scores = scorer.score_multiple_resumes(resumes, job)
        assert len(scores) == 3
        assert all(0 <= score <= 1 for _, score in scores)


class TestRankingModel:
    """Test ranking models"""
    
    def test_logistic_regression_training(self):
        """Test logistic regression model"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model = RankingModel(model_type='logistic_regression')
        model.train(X, y)
        
        assert model.is_trained
    
    def test_model_prediction(self):
        """Test model prediction"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model = RankingModel(model_type='gradient_boosting')
        model.train(X, y)
        
        X_test = np.random.rand(10, 10)
        predictions = model.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model = RankingModel()
        model.train(X[:40], y[:40])
        
        metrics = model.evaluate(X[40:], y[40:])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
API module for Resume Screening System
"""

from flask import Flask, request, jsonify
import numpy as np
from typing import List, Dict
import logging

from resume_screening.ranker import ResumeRanker
from resume_screening.embeddings import BERTEmbedder
from resume_screening.similarity import SimilarityScorer
from resume_screening.utils import get_logger

# Initialize Flask app
app = Flask(__name__)
logger = get_logger(__name__)

# Global models (initialized once)
resume_ranker = None
bert_embedder = None
similarity_scorer = None


def initialize_models():
    """Initialize models on app startup"""
    global resume_ranker, bert_embedder, similarity_scorer
    
    logger.info("Initializing models...")
    
    bert_embedder = BERTEmbedder()
    similarity_scorer = SimilarityScorer(bert_embedder)
    resume_ranker = ResumeRanker(embedder_type='bert', 
                                 model_type='gradient_boosting')
    
    logger.info("Models initialized successfully")


@app.before_request
def before_request():
    """Initialize models on first request"""
    global resume_ranker
    if resume_ranker is None:
        initialize_models()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Resume Screening API'
    })


@app.route('/rank', methods=['POST'])
def rank_resumes():
    """
    Rank resumes against a job description
    
    Request body:
    {
        "resumes": ["resume1 text", "resume2 text", ...],
        "job_description": "job description text",
        "top_k": 5  (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        resumes = data.get('resumes')
        job_description = data.get('job_description')
        top_k = data.get('top_k', len(resumes) if resumes else 5)
        
        # Validate input
        if not resumes or not isinstance(resumes, list):
            return jsonify({'error': 'resumes must be a non-empty list'}), 400
        
        if not job_description or not isinstance(job_description, str):
            return jsonify({'error': 'job_description must be a non-empty string'}), 400
        
        # Rank resumes
        rankings = resume_ranker.rank_resumes(resumes, job_description)
        
        # Return top k
        top_rankings = rankings[:top_k]
        
        results = []
        for rank, (idx, score) in enumerate(top_rankings, 1):
            results.append({
                'rank': rank,
                'resume_index': int(idx),
                'score': float(score),
                'match_percentage': float(score * 100)
            })
        
        return jsonify({
            'status': 'success',
            'total_resumes': len(resumes),
            'rankings': results
        })
    
    except Exception as e:
        logger.error(f"Error in rank endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/score', methods=['POST'])
def score_resume():
    """
    Compute similarity score between a resume and job description
    
    Request body:
    {
        "resume": "resume text",
        "job_description": "job description text"
    }
    """
    try:
        data = request.get_json()
        
        resume = data.get('resume')
        job_description = data.get('job_description')
        
        if not resume or not job_description:
            return jsonify({'error': 'resume and job_description are required'}), 400
        
        # Compute score
        score = similarity_scorer.score_resume(resume, job_description, 
                                              metric='cosine')
        
        return jsonify({
            'status': 'success',
            'score': float(score),
            'match_percentage': float(score * 100)
        })
    
    except Exception as e:
        logger.error(f"Error in score endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their status"""
    try:
        models_info = {
            'embedder': {
                'type': 'BERT',
                'model': 'all-MiniLM-L6-v2',
                'status': 'loaded' if bert_embedder else 'not loaded'
            },
            'ranker': {
                'type': 'Gradient Boosting',
                'status': 'loaded' if resume_ranker else 'not loaded'
            },
            'similarity_scorer': {
                'type': 'Cosine Similarity',
                'status': 'loaded' if similarity_scorer else 'not loaded'
            }
        }
        
        return jsonify({
            'status': 'success',
            'models': models_info
        })
    
    except Exception as e:
        logger.error(f"Error in models endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_rank', methods=['POST'])
def batch_rank():
    """
    Rank resumes against multiple job descriptions
    
    Request body:
    {
        "resumes": ["resume1", "resume2", ...],
        "job_descriptions": [
            {"title": "Position 1", "description": "..."},
            {"title": "Position 2", "description": "..."}
        ]
    }
    """
    try:
        data = request.get_json()
        
        resumes = data.get('resumes')
        job_descriptions = data.get('job_descriptions')
        
        if not resumes or not job_descriptions:
            return jsonify({
                'error': 'resumes and job_descriptions are required'
            }), 400
        
        results = []
        
        for job_idx, job_info in enumerate(job_descriptions):
            job_title = job_info.get('title', f'Job {job_idx}')
            job_desc = job_info.get('description')
            
            if not job_desc:
                continue
            
            rankings = resume_ranker.rank_resumes(resumes, job_desc)
            
            results.append({
                'job_index': job_idx,
                'job_title': job_title,
                'top_match': {
                    'resume_index': int(rankings[0][0]),
                    'score': float(rankings[0][1])
                }
            })
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error in batch_rank endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'POST /rank',
            'POST /score',
            'POST /batch_rank',
            'GET /models',
            'GET /health'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Resume Screening API")
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)

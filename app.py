"""
Web Application for Resume Screening System
Serves the frontend and API endpoints
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from typing import List, Dict

from resume_screening.embeddings import TFIDFEmbedder
from resume_screening.similarity import SimilarityScorer
from resume_screening.preprocessor import TextPreprocessor

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Global models
embedder = None
scorer = None
preprocessor = None


def initialize_models():
    """Initialize models on app startup"""
    global embedder, scorer, preprocessor
    print("Initializing models...")
    preprocessor = TextPreprocessor()
    print("Models initialized successfully!")


@app.before_request
def before_request():
    """Initialize models on first request"""
    global preprocessor
    if preprocessor is None:
        initialize_models()


# =============================================================================
# Frontend Routes
# =============================================================================

@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


# =============================================================================
# API Routes
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Resume Screening API',
        'version': '1.0.0'
    })


@app.route('/api/rank', methods=['POST'])
def rank_resumes():
    """
    Rank resumes against a job description
    
    Request body:
    {
        "resumes": ["resume1 text", "resume2 text", ...],
        "job_description": "job description text"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        resumes = data.get('resumes', [])
        job_description = data.get('job_description', '')
        
        # Validate input
        if not resumes or not isinstance(resumes, list):
            return jsonify({'success': False, 'error': 'resumes must be a non-empty list'}), 400
        
        if not job_description:
            return jsonify({'success': False, 'error': 'job_description is required'}), 400
        
        # Initialize TF-IDF embedder with all documents
        embedder = TFIDFEmbedder()
        all_docs = resumes + [job_description]
        embedder.train(all_docs)
        
        # Create scorer
        scorer = SimilarityScorer(embedder)
        
        # Score each resume
        results = []
        details = {}
        
        for i, resume in enumerate(resumes):
            score = scorer.score_resume(resume, job_description)
            results.append({
                'resume_index': i,
                'score': float(score)
            })
            
            # Extract details
            skills = TextPreprocessor.extract_skills(resume)
            emails = TextPreprocessor.extract_emails(resume)
            phones = TextPreprocessor.extract_phones(resume)
            
            details[i] = {
                'skills': skills,
                'emails': emails,
                'phones': phones
            }
        
        # Sort by score descending
        rankings = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'success': True,
            'rankings': rankings,
            'details': details,
            'total_resumes': len(resumes)
        })
        
    except Exception as e:
        print(f"Error ranking resumes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@app.route('/api/extract-skills', methods=['POST'])
def extract_skills():
    """
    Extract skills from text
    
    Request body:
    {
        "text": "resume or job description text"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'text is required'}), 400
        
        skills = TextPreprocessor.extract_skills(text)
        emails = TextPreprocessor.extract_emails(text)
        phones = TextPreprocessor.extract_phones(text)
        
        return jsonify({
            'success': True,
            'skills': skills,
            'emails': emails,
            'phones': phones
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_text():
    """
    Preprocess text (tokenize, clean, etc.)
    
    Request body:
    {
        "text": "text to preprocess"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'text is required'}), 400
        
        tokens = preprocessor.process(text)
        
        return jsonify({
            'success': True,
            'tokens': tokens,
            'token_count': len(tokens)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Resume Screening System - Web Application")
    print("="*60)
    print("\nStarting server...")
    print("Open http://localhost:8080 in your browser")
    print("="*60 + "\n")
    
    # Initialize models at startup
    initialize_models()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)

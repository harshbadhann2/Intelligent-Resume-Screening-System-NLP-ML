"""
Web Application for Resume Screening System
Serves the frontend and API endpoints
"""

import os
import sys
import io
import re
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from typing import List, Dict
import requests

# PDF and Document processing
from PyPDF2 import PdfReader
from docx import Document

from resume_screening.embeddings import TFIDFEmbedder
from resume_screening.similarity import SimilarityScorer
from resume_screening.preprocessor import TextPreprocessor


# =============================================================================
# Document Processing Functions
# =============================================================================

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""


def extract_text_from_docx(file_stream):
    """Extract text from a DOCX file"""
    try:
        doc = Document(file_stream)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""


def extract_text_from_url(url):
    """Extract text from a URL (supports PDF, DOCX links or plain text)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Handle PDF
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            return extract_text_from_pdf(io.BytesIO(response.content))
        
        # Handle DOCX
        elif 'document' in content_type or url.lower().endswith('.docx'):
            return extract_text_from_docx(io.BytesIO(response.content))
        
        # Handle DOC (older format) - try as plain text
        elif url.lower().endswith('.doc'):
            return response.text[:10000]  # Limit size
        
        # Plain text / HTML - extract text content
        else:
            # Simple HTML tag removal
            text = re.sub(r'<[^>]+>', ' ', response.text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()[:10000]
            
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

# Global models
embedder = None
scorer = None
preprocessor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        'version': '2.0.0',
        'features': ['pdf', 'docx', 'url', 'text']
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload and extract text from PDF, DOCX, or DOC files
    Returns extracted text content
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        filename = file.filename.lower()
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file.stream)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file.stream)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8', errors='ignore')
        else:
            # For .doc files, try to read as binary/text
            text = file.read().decode('utf-8', errors='ignore')
        
        if not text:
            return jsonify({
                'success': False, 
                'error': 'Could not extract text from file. Please ensure the file is not encrypted or corrupted.'
            }), 400
        
        # Also extract skills for preview
        skills = TextPreprocessor.extract_skills(text)
        emails = TextPreprocessor.extract_emails(text)
        phones = TextPreprocessor.extract_phones(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': file.filename,
            'preview': {
                'skills': skills[:10],
                'emails': emails,
                'phones': phones,
                'char_count': len(text),
                'word_count': len(text.split())
            }
        })
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fetch-url', methods=['POST'])
def fetch_url():
    """
    Fetch and extract text from a URL
    Supports PDF, DOCX links and web pages
    """
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        text = extract_text_from_url(url)
        
        if not text:
            return jsonify({
                'success': False, 
                'error': 'Could not extract text from URL. Check the link and try again.'
            }), 400
        
        # Extract skills for preview
        skills = TextPreprocessor.extract_skills(text)
        emails = TextPreprocessor.extract_emails(text)
        phones = TextPreprocessor.extract_phones(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'url': url,
            'preview': {
                'skills': skills[:10],
                'emails': emails,
                'phones': phones,
                'char_count': len(text),
                'word_count': len(text.split())
            }
        })
        
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


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

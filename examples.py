"""
Example: Complete Resume Screening Pipeline
Works with Python 3.14 using TF-IDF embeddings (no BERT/Word2Vec dependencies)
"""

import sys
sys.path.insert(0, '.')

from resume_screening import TextPreprocessor, TFIDFEmbedder, SimilarityScorer

# ============================================================================
# EXAMPLE 1: Simple Resume Ranking with TF-IDF
# ============================================================================

def example_basic_ranking():
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Resume Ranking with TF-IDF")
    print("="*70)
    
    # Sample data
    resumes = [
        """
        John Smith
        Senior Python Developer
        
        Experience:
        - 7 years Python development
        - Django framework expertise
        - AWS cloud architecture
        - Machine learning with TensorFlow
        
        Skills: Python, Django, FastAPI, TensorFlow, PyTorch, AWS, Docker
        """,
        
        """
        Jane Doe
        Java Backend Engineer
        
        Experience:
        - 5 years Java development
        - Spring Boot expertise
        - Microservices architecture
        - Kubernetes orchestration
        
        Skills: Java, Spring, Kubernetes, PostgreSQL, Kafka
        """,
        
        """
        Bob Wilson
        Full Stack Developer
        
        Experience:
        - 6 years web development
        - React and Node.js
        - Database design
        - Team leadership
        
        Skills: JavaScript, React, Node.js, MongoDB, CSS
        """
    ]
    
    job_description = """
    Senior Python Developer - Machine Learning
    
    We are looking for an experienced Python developer with:
    - 5+ years Python development experience
    - Strong machine learning background (TensorFlow/PyTorch)
    - Django or FastAPI experience
    - AWS cloud expertise
    - Experience building production ML systems
    
    Responsibilities:
    - Design and implement ML models
    - Build Python APIs and services
    - Optimize model performance
    - Collaborate with data science team
    """
    
    # Initialize TF-IDF embedder
    embedder = TFIDFEmbedder()
    all_docs = resumes + [job_description]
    embedder.train(all_docs)
    
    # Score each resume
    scorer = SimilarityScorer(embedder)
    
    results = []
    for i, resume in enumerate(resumes):
        score = scorer.score_resume(resume, job_description)
        results.append((i, score))
    
    # Sort by score
    rankings = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nRanking Results:")
    print("-" * 70)
    for rank, (resume_idx, score) in enumerate(rankings, 1):
        print(f"{rank}. Resume {resume_idx}: {score:.4f} ({score*100:.2f}% match)")
    
    return rankings


# ============================================================================
# EXAMPLE 2: Text Preprocessing and Skill Extraction
# ============================================================================

def example_preprocessing():
    print("\n" + "="*70)
    print("EXAMPLE 2: Text Preprocessing and Information Extraction")
    print("="*70)
    
    resume = """
    Dr. Alice Johnson
    Email: alice.johnson@email.com
    Phone: (555) 123-4567
    
    Summary:
    Experienced Machine Learning Engineer with 8 years in NLP and Computer Vision.
    Expert in Python, TensorFlow, and PyTorch. Published researcher with 15+ papers.
    
    Skills:
    - Python, Rust, Java
    - TensorFlow, PyTorch, Keras
    - NLP (BERT, GPT, Transformers)
    - Computer Vision (CNN, Object Detection)
    - AWS, GCP, Docker, Kubernetes
    - Apache Spark, Hadoop
    """
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Extract skills
    skills = TextPreprocessor.extract_skills(resume)
    print(f"\nExtracted Skills: {skills}")
    
    # Extract contact info
    emails = TextPreprocessor.extract_emails(resume)
    phones = TextPreprocessor.extract_phones(resume)
    print(f"Emails: {emails}")
    print(f"Phones: {phones}")
    
    # Preprocess text
    tokens = preprocessor.process(resume)
    print(f"\nTokenized ({len(tokens)} tokens): {tokens[:20]}...")


# ============================================================================
# EXAMPLE 3: TF-IDF Similarity Analysis
# ============================================================================

def example_tfidf_similarity():
    print("\n" + "="*70)
    print("EXAMPLE 3: TF-IDF Similarity Analysis")
    print("="*70)
    
    documents = [
        "Machine learning and deep learning for data science",
        "Natural language processing with transformers and BERT",
        "Computer vision and image recognition systems",
        "Data science and predictive analytics"
    ]
    
    print("\nTraining TF-IDF embedder...")
    
    # TF-IDF
    tfidf = TFIDFEmbedder()
    tfidf.train(documents)
    tfidf_emb = tfidf.embed(documents[0])
    print(f"TF-IDF embedding: {tfidf_emb.shape[0]} dimensions")
    
    # Compare similarity
    query = "machine learning and deep neural networks"
    
    scorer = SimilarityScorer(tfidf)
    
    print(f"\nSimilarity to query: '{query}'")
    print("-" * 50)
    for doc in documents:
        score = scorer.score_resume(query, doc)
        print(f"  {score:.4f} - {doc[:50]}...")


# ============================================================================
# EXAMPLE 4: Batch Processing Multiple Jobs
# ============================================================================

def example_batch_processing():
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing Multiple Job Descriptions")
    print("="*70)
    
    resumes = [
        "Python developer with 5 years experience in machine learning and TensorFlow",
        "Java engineer specializing in microservices and spring boot architecture",
        "Full stack developer with React and Node.js expertise for web applications"
    ]
    
    jobs = [
        ("Python ML Engineer", "Looking for Python developer with ML experience and TensorFlow"),
        ("Java Backend Dev", "Seeking Java expert for microservices and Spring Boot"),
        ("Frontend Developer", "Need React expert for UI development and web apps")
    ]
    
    # Train on all documents
    embedder = TFIDFEmbedder()
    all_docs = resumes + [job[1] for job in jobs]
    embedder.train(all_docs)
    
    scorer = SimilarityScorer(embedder)
    
    print("\nRanking Results:")
    print("-" * 70)
    
    for job_title, job_desc in jobs:
        results = []
        for idx, resume in enumerate(resumes):
            score = scorer.score_resume(resume, job_desc)
            results.append((idx, score))
        
        rankings = sorted(results, key=lambda x: x[1], reverse=True)
        
        print(f"\n{job_title}:")
        for rank, (idx, score) in enumerate(rankings, 1):
            print(f"  {rank}. Resume {idx}: {score:.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("RESUME SCREENING SYSTEM - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Run examples
        example_basic_ranking()
        example_preprocessing()
        example_tfidf_similarity()
        example_batch_processing()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        import traceback
        print(f"\nError running examples: {e}")
        traceback.print_exc()
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

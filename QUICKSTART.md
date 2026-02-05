# Quick Start Guide - Resume Screening System

## Installation

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Resume Ranking

```python
from resume_screening.ranker import ResumeRanker

# Initialize ranker with BERT embeddings and Gradient Boosting
ranker = ResumeRanker(embedder_type='bert', model_type='gradient_boosting')

# Load resumes and job description
resumes = [
    "I am a Python developer with 5 years experience...",
    "Java expert with machine learning background...",
    "Full-stack developer proficient in JavaScript..."
]

job_description = """
Senior Python Developer
Requirements:
- 5+ years Python experience
- Django or Flask expertise
- Machine learning background
- AWS cloud experience
"""

# Rank resumes
rankings = ranker.rank_resumes(resumes, job_description)

# Print results
for rank, (resume_idx, score) in enumerate(rankings, 1):
    print(f"{rank}. Resume {resume_idx}: {score:.4f} ({score*100:.2f}%)")
```

### Similarity Scoring

```python
from resume_screening.similarity import SimilarityScorer
from resume_screening.embeddings import BERTEmbedder

# Initialize scorer
embedder = BERTEmbedder()
scorer = SimilarityScorer(embedder)

# Score single pair
resume = "Python developer with ML experience"
job = "Looking for Python expert with machine learning"
score = scorer.score_resume(resume, job)

print(f"Match score: {score:.4f}")
```

### Text Preprocessing

```python
from resume_screening.preprocessor import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor(remove_stopwords=True, use_lemmatization=True)

# Preprocess text
text = "I am an experienced Software Developer with strong Python skills..."
tokens = preprocessor.process(text)

# Extract skills
skills = TextPreprocessor.extract_skills(text)

# Extract contact info
emails = TextPreprocessor.extract_emails(text)
phones = TextPreprocessor.extract_phones(text)
```

### Using Embeddings

```python
from resume_screening.embeddings import TFIDFEmbedder, Word2VecEmbedder, BERTEmbedder

# TF-IDF
tfidf = TFIDFEmbedder()
tfidf.train(documents)
embeddings = tfidf.embed("sample text")

# Word2Vec
w2v = Word2VecEmbedder(vector_size=300)
w2v.train(documents)
embeddings = w2v.embed("sample text")

# BERT
bert = BERTEmbedder(model_name='all-MiniLM-L6-v2')
embeddings = bert.embed("sample text")
similarity = bert.similarity("text1", "text2")
```

## Jupyter Notebooks

Run the notebooks for detailed exploration and model training:

```bash
jupyter notebook notebooks/
```

### Available Notebooks:
1. **01_eda.ipynb** - Exploratory data analysis of resumes and job descriptions
2. **02_embeddings.ipynb** - Generate and compare TF-IDF, Word2Vec, BERT embeddings
3. **03_similarity.ipynb** - Compute similarity scores using different metrics
4. **04_ranking.ipynb** - Train and evaluate classification models

## API Usage

### Start API Server
```bash
python -m resume_screening.api
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Rank Resumes
```bash
curl -X POST http://localhost:5000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "resumes": ["resume text 1", "resume text 2"],
    "job_description": "job description text",
    "top_k": 5
  }'
```

#### Score Single Pair
```bash
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "resume text",
    "job_description": "job description"
  }'
```

#### List Models
```bash
curl http://localhost:5000/models
```

#### Batch Ranking
```bash
curl -X POST http://localhost:5000/batch_rank \
  -H "Content-Type: application/json" \
  -d '{
    "resumes": ["resume1", "resume2"],
    "job_descriptions": [
      {"title": "Job 1", "description": "..."},
      {"title": "Job 2", "description": "..."}
    ]
  }'
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_resume_screening.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=resume_screening
```

## Project Structure

```
├── resume_screening/          # Main package
│   ├── __init__.py
│   ├── preprocessor.py        # Text preprocessing
│   ├── embeddings.py          # TF-IDF, Word2Vec, BERT
│   ├── similarity.py          # Similarity scoring
│   ├── ranker.py              # Ranking models
│   ├── data_loader.py         # Dataset handling
│   ├── utils.py               # Utility functions
│   └── api.py                 # Flask API
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_similarity.ipynb
│   └── 04_ranking.ipynb
├── data/                      # Data directory
│   ├── raw/                   # Raw datasets
│   └── processed/             # Processed datasets
├── models/                    # Trained models
│   ├── tfidf_model/
│   ├── word2vec_model/
│   ├── bert_model/
│   └── ranking_model/
├── tests/                     # Unit tests
│   └── test_resume_screening.py
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Performance Tips

1. **Use BERT for best accuracy** - Provides semantic understanding but slower (~0.5s per pair)
2. **Use TF-IDF for speed** - Fast vectorization, good for large-scale screening
3. **Use Word2Vec for balance** - Good accuracy with moderate speed
4. **Batch processing** - Process multiple resumes against one job for efficiency
5. **Model caching** - Load models once and reuse for multiple queries

## Troubleshooting

### BERT Model Download
If BERT takes long on first use, it's downloading the model (~400MB). Subsequent runs will be fast.

### Memory Issues
Reduce batch size or use lighter models (TF-IDF, MiniLM instead of large BERT)

### Import Errors
Ensure you're in the correct virtual environment and all dependencies are installed

## Contributing

1. Add new features in respective modules
2. Write unit tests in `tests/` directory
3. Update notebooks with examples
4. Follow PEP 8 style guidelines

## References

- BERT: https://arxiv.org/abs/1810.04805
- Word2Vec: https://arxiv.org/abs/1301.3781
- Scikit-learn: https://scikit-learn.org/
- Sentence Transformers: https://www.sbert.net/

---

For detailed documentation, see README.md

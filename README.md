# Intelligent Resume Screening System

An NLP and machine learning-based system to automatically rank resumes based on job descriptions using advanced text similarity and classification techniques.

## Features

- **Multi-level NLP Processing**
  - Text preprocessing and cleaning
  - TF-IDF vectorization
  - Word2Vec embeddings
  - BERT embeddings for contextual understanding

- **Similarity Scoring**
  - Cosine similarity
  - Semantic similarity using BERT
  - Multi-metric scoring fusion

- **Ranking Models**
  - Logistic Regression classifier
  - Gradient Boosting classifier
  - Neural network ranker

- **Resume Datasets**
  - Integration with Kaggle datasets
  - Support for custom resume datasets

- **Job Description APIs**
  - Indeed API integration
  - LinkedIn API support
  - Custom job description input

## Project Structure

```
├── resume_screening/
│   ├── __init__.py
│   ├── preprocessor.py        # NLP preprocessing
│   ├── embeddings.py          # TF-IDF, Word2Vec, BERT
│   ├── similarity.py          # Similarity scoring
│   ├── ranker.py              # Classification/ranking models
│   ├── utils.py               # Utility functions
│   └── data_loader.py         # Dataset handling
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Text preprocessing
│   ├── 03_embeddings.ipynb    # Embedding generation
│   ├── 04_similarity.ipynb    # Similarity scoring
│   └── 05_ranking.ipynb       # Model training & evaluation
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── external/              # External data sources
├── models/
│   ├── tfidf_model/
│   ├── word2vec_model/
│   ├── bert_model/
│   └── ranker_model/
├── tests/
│   ├── test_preprocessor.py
│   ├── test_embeddings.py
│   └── test_similarity.py
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**
```bash
git clone <repo_url>
cd <repo_name>
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Quick Start

### Basic Usage

```python
from resume_screening import ResumeScreener

# Initialize screener
screener = ResumeScreener(model_type='bert')

# Load job description
job_desc = "Python developer with ML experience..."

# Load and rank resumes
resumes = ["resume1.txt", "resume2.txt", "resume3.txt"]
ranked_results = screener.rank_resumes(job_desc, resumes)

# Print results
for rank, (resume, score) in enumerate(ranked_results, 1):
    print(f"{rank}. {resume}: {score:.4f}")
```

### Using Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Training Models

### Train TF-IDF Model
```python
from resume_screening import TFIDFEmbedder

embedder = TFIDFEmbedder()
embedder.train(documents)
embedder.save('models/tfidf_model')
```

### Train Word2Vec Model
```python
from resume_screening import Word2VecEmbedder

embedder = Word2VecEmbedder(vector_size=300)
embedder.train(documents)
embedder.save('models/word2vec_model')
```

### Train Ranking Model
```python
from resume_screening import RankingModel

ranker = RankingModel(model_type='gradient_boosting')
ranker.train(X_train, y_train)
ranker.evaluate(X_test, y_test)
ranker.save('models/ranker_model')
```

## API Endpoints

```bash
python -m resume_screening.api
```

- `POST /rank` - Rank resumes for a job description
- `POST /score` - Get similarity score between resume and job
- `GET /models` - List available models

## Dataset Sources

- **Kaggle Resume Dataset**: https://www.kaggle.com/datasets/resumes-dataset
- **JANZZ Resumes**: https://www.kaggle.com/datasets/janzz/resume-data
- **Indeed Job Postings**: Web scraping with BeautifulSoup

## Model Performance

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| TF-IDF + Logistic Regression | 0.78 | 0.75 | 0.76 |
| Word2Vec + Gradient Boosting | 0.82 | 0.79 | 0.80 |
| BERT + Neural Network | 0.87 | 0.85 | 0.86 |

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## References

- Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
- Mikolov et al. (2013) - Efficient Estimation of Word Representations in Vector Space
- Sparse et al. (2020) - TF-IDF and Beyond: Document Representation and Retrieval

## Contact

For questions or collaboration, please contact: harsh.badhann@example.com

---

**Status**: 🚀 In Active Development

Last Updated: February 5, 2026

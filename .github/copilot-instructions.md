# Resume Screening System - Project Instructions

## Project Overview
Intelligent Resume Screening System using NLP and Machine Learning to automatically rank resumes based on job descriptions.

### Tech Stack
- **Language**: Python 3.9+
- **Core Libraries**: 
  - NLP: NLTK, Transformers (BERT), Gensim (Word2Vec)
  - ML: scikit-learn, torch
  - Data: pandas, numpy, scipy
- **Development**: Jupyter, pytest

### Key Components
1. **NLP Preprocessing**: Text cleaning, tokenization, lemmatization
2. **Embeddings**: TF-IDF, Word2Vec, BERT
3. **Similarity Scoring**: Cosine, semantic similarity
4. **Ranking Models**: Logistic Regression, Gradient Boosting, Neural Networks

## Setup Progress

- [x] Project scaffolding complete
- [x] Directory structure created
- [x] requirements.txt generated
- [x] README.md created
- [ ] Core modules implementation
- [ ] Jupyter notebooks created
- [ ] Model training pipelines
- [ ] Unit tests
- [ ] API endpoints

## Development Workflow

1. **Data Preparation**: Download/prepare datasets in `data/raw/`
2. **Exploration**: Use notebooks in `notebooks/` for EDA
3. **Model Development**: Implement in `resume_screening/` modules
4. **Testing**: Run tests in `tests/`
5. **Deployment**: API endpoints in `resume_screening/api.py`

## Important Notes

- Models are saved in `models/` directory
- Processed datasets cached in `data/processed/`
- Use virtual environment for dependency isolation
- BERT model downloads on first use (~400MB)

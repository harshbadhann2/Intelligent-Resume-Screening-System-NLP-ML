"""
Text Preprocessing Module for Resume and Job Description Processing
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """
    Comprehensive text preprocessing for resumes and job descriptions
    
    Handles:
    - Text cleaning and normalization
    - Tokenization
    - Stopword removal
    - Lemmatization/Stemming
    - Entity extraction
    """
    
    def __init__(self, remove_stopwords: bool = True, use_lemmatization: bool = True):
        """
        Initialize preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            use_lemmatization: Use lemmatization (True) or stemming (False)
        """
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Remove special characters, URLs, emails, and normalize whitespace"""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s\-\.]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def tokenize(self, text: str, sentence_level: bool = False) -> List[str]:
        """
        Tokenize text into words or sentences
        
        Args:
            text: Input text
            sentence_level: If True, return sentence tokens; else word tokens
            
        Returns:
            List of tokens
        """
        if sentence_level:
            return sent_tokenize(text)
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common English stopwords"""
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if not self.use_lemmatization:
            return [self.stemmer.stem(token) for token in tokens]
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text: str, remove_stopwords: bool = None, 
                lemmatize: bool = None) -> List[str]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            remove_stopwords: Override instance setting
            lemmatize: Override instance setting
            
        Returns:
            List of processed tokens
        """
        # Use instance settings if not overridden
        remove_sw = remove_stopwords if remove_stopwords is not None else self.remove_stopwords
        do_lemma = lemmatize if lemmatize is not None else self.use_lemmatization
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        if remove_sw:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        
        # Lemmatize/Stem
        if do_lemma:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        else:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Remove empty tokens
        tokens = [t for t in tokens if t and len(t) > 1]
        
        return tokens
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        cleaned = self.clean_text(text)
        return sent_tokenize(cleaned)
    
    @staticmethod
    def extract_skills(text: str, skill_list: List[str] = None) -> List[str]:
        """
        Extract technical skills from resume/job description
        
        Args:
            text: Input text
            skill_list: List of known skills to search for
            
        Returns:
            List of found skills
        """
        # Default common tech skills
        if skill_list is None:
            skill_list = [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go',
                'machine learning', 'deep learning', 'nlp', 'computer vision',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn',
                'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes',
                'react', 'angular', 'vue', 'nodejs', 'express',
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
                'git', 'linux', 'unix', 'windows', 'macos',
                'agile', 'scrum', 'jira', 'confluence',
                'rest api', 'graphql', 'json', 'xml',
                'html', 'css', 'bootstrap', 'tailwind'
            ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_list:
            if skill in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def extract_phones(text: str) -> List[str]:
        """Extract phone numbers from text"""
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        return re.findall(phone_pattern, text)


class ResumeParser:
    """Parse and structure resume information"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def parse(self, text: str) -> dict:
        """
        Parse resume and extract key information
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with structured resume data
        """
        resume_data = {
            'raw_text': text,
            'cleaned_text': self.preprocessor.clean_text(text),
            'tokens': self.preprocessor.process(text),
            'sentences': self.preprocessor.extract_sentences(text),
            'skills': self.preprocessor.extract_skills(text),
            'emails': self.preprocessor.extract_emails(text),
            'phones': self.preprocessor.extract_phones(text),
        }
        
        return resume_data

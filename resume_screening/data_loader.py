"""
Data Loading and Dataset Management
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
import json
from pathlib import Path


class DataLoader:
    """Load and manage datasets for resume screening"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize data loader
        
        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV file
        
        Args:
            filename: Name of CSV file in raw directory
            
        Returns:
            DataFrame
        """
        filepath = os.path.join(self.raw_dir, filename)
        return pd.read_csv(filepath)
    
    def load_json(self, filename: str) -> dict:
        """
        Load JSON file
        
        Args:
            filename: Name of JSON file in raw directory
            
        Returns:
            Dictionary
        """
        filepath = os.path.join(self.raw_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_text_files(self, directory: str) -> List[Tuple[str, str]]:
        """
        Load all text files from directory
        
        Args:
            directory: Directory path
            
        Returns:
            List of (filename, content) tuples
        """
        texts = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    texts.append((filename, content))
        return texts
    
    def load_resumes(self, resume_dir: str = None) -> List[Tuple[str, str]]:
        """
        Load resume files
        
        Args:
            resume_dir: Directory containing resumes (default: raw/resumes)
            
        Returns:
            List of (filename, content) tuples
        """
        if resume_dir is None:
            resume_dir = os.path.join(self.raw_dir, 'resumes')
        
        return self.load_text_files(resume_dir)
    
    def load_job_descriptions(self, job_dir: str = None) -> List[Tuple[str, str]]:
        """
        Load job description files
        
        Args:
            job_dir: Directory containing job descriptions (default: raw/jobs)
            
        Returns:
            List of (filename, content) tuples
        """
        if job_dir is None:
            job_dir = os.path.join(self.raw_dir, 'jobs')
        
        return self.load_text_files(job_dir)
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """
        Save processed data
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.processed_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data
        
        Args:
            filename: Filename in processed directory
            
        Returns:
            DataFrame
        """
        filepath = os.path.join(self.processed_dir, filename)
        return pd.read_csv(filepath)
    
    @staticmethod
    def create_training_dataset(resumes: List[str], 
                               job_descriptions: List[str],
                               labels: List[int]) -> pd.DataFrame:
        """
        Create training dataset
        
        Args:
            resumes: List of resume texts
            job_descriptions: List of job description texts
            labels: List of binary labels
            
        Returns:
            DataFrame with (resume, job_description, label)
        """
        return pd.DataFrame({
            'resume': resumes,
            'job_description': job_descriptions,
            'label': labels
        })
    
    @staticmethod
    def kaggle_resume_dataset() -> str:
        """
        Information about Kaggle resume dataset
        
        Returns:
            Dataset description
        """
        return """
        Kaggle Resume Dataset
        - Source: https://www.kaggle.com/datasets/resumes-dataset
        - Contains: ~2000 resumes with categories
        - Categories: IT, HR, Finance, etc.
        
        Download instructions:
        1. Visit Kaggle dataset link
        2. Download 'Resume.csv'
        3. Place in data/raw/ directory
        """
    
    @staticmethod
    def janzz_dataset() -> str:
        """
        Information about JANZZ resume dataset
        
        Returns:
            Dataset description
        """
        return """
        JANZZ Resume Dataset
        - Source: https://www.kaggle.com/datasets/janzz/resume-data
        - Contains: Job-resume pairs
        - Format: CSV with resume and job text
        
        Download instructions:
        1. Visit Kaggle dataset link
        2. Download data files
        3. Place in data/raw/ directory
        """


class JobScraper:
    """
    Utilities for scraping job descriptions
    (Note: API integration would require actual credentials)
    """
    
    @staticmethod
    def indeed_api_setup() -> str:
        """
        Instructions for Indeed API setup
        
        Returns:
            Setup instructions
        """
        return """
        Indeed API Setup
        1. Create Indeed publisher account
        2. Get API credentials
        3. Store in .env file:
           INDEED_API_KEY=your_key
           INDEED_API_SECRET=your_secret
        4. Use requests library to query API
        """
    
    @staticmethod
    def linkedin_api_setup() -> str:
        """
        Instructions for LinkedIn API setup
        
        Returns:
            Setup instructions
        """
        return """
        LinkedIn API Setup
        1. Create LinkedIn developer app
        2. Get credentials
        3. Store in .env file:
           LINKEDIN_API_KEY=your_key
        4. Use linkedin_api library
        """
    
    @staticmethod
    def sample_job_descriptions() -> Dict[str, str]:
        """
        Sample job descriptions for testing
        
        Returns:
            Dictionary of sample jobs
        """
        return {
            'python_developer': """
            Python Developer Position
            
            Requirements:
            - 3+ years Python experience
            - Strong knowledge of Django or Flask
            - SQL and NoSQL databases
            - REST API development
            - Unit testing and TDD
            - Git version control
            - Agile/Scrum experience
            
            Nice to have:
            - Machine learning experience
            - AWS or cloud experience
            - Docker containerization
            - CI/CD pipelines
            """,
            
            'ml_engineer': """
            Machine Learning Engineer
            
            Requirements:
            - 2+ years ML/AI experience
            - Python, TensorFlow, PyTorch
            - Deep learning fundamentals
            - NLP or Computer Vision
            - SQL and Big Data tools
            - Model deployment experience
            - Statistical analysis
            
            Responsibilities:
            - Design ML models
            - Data preprocessing and feature engineering
            - Model training and evaluation
            - Deployment and monitoring
            - Collaborate with data scientists
            """,
            
            'data_scientist': """
            Data Scientist
            
            Requirements:
            - PhD or Master's in related field
            - Python, R proficiency
            - Statistical analysis
            - Machine learning algorithms
            - Data visualization
            - SQL and database knowledge
            - Communication skills
            
            Responsibilities:
            - Analyze large datasets
            - Build predictive models
            - A/B testing
            - Create dashboards
            - Communicate insights to stakeholders
            """
        }


class SyntheticDataGenerator:
    """Generate synthetic training data for testing"""
    
    @staticmethod
    def generate_matched_pairs(n_pairs: int = 100) -> Tuple[List[str], List[str], List[int]]:
        """
        Generate synthetic resume-job matched pairs
        
        Args:
            n_pairs: Number of pairs to generate
            
        Returns:
            (resumes, job_descriptions, labels)
        """
        templates = {
            'resume': [
                "I am an experienced {skills} professional with {years} years in {domain}. "
                "Proficient in {tech}. Led projects in {area}.",
                "{years} years {domain} expert. Skills: {skills}, {tech}. "
                "Proven track record in {area}.",
                "Senior {domain} specialist with {years}+ experience. "
                "Expertise: {skills}. Specialized in {area}."
            ],
            'job': [
                "Looking for {domain} expert with {years}+ years experience. "
                "Required: {skills}, {tech}. Will work on {area}.",
                "{domain} position. Need professional with: {skills}, {tech}, {years} years. "
                "Focus on {area}.",
                "Senior {domain} role. Requirements: {years}+ exp, {skills}, {tech}. "
                "Project involves {area}."
            ]
        }
        
        domains = ['Python', 'Data Science', 'ML Engineering', 'DevOps', 'Full Stack']
        skills = ['problem-solving', 'communication', 'leadership', 'analytical thinking']
        techs = ['TensorFlow', 'Django', 'Kubernetes', 'PostgreSQL', 'AWS']
        areas = ['distributed systems', 'NLP', 'CV', 'cloud infrastructure', 'analytics']
        
        resumes = []
        jobs = []
        labels = []
        
        for i in range(n_pairs):
            domain = np.random.choice(domains)
            skill = np.random.choice(skills)
            tech = np.random.choice(techs)
            area = np.random.choice(areas)
            years = np.random.randint(2, 15)
            
            resume = np.random.choice(templates['resume']).format(
                skills=skill, years=years, domain=domain, tech=tech, area=area
            )
            job = np.random.choice(templates['job']).format(
                skills=skill, years=years-1, domain=domain, tech=tech, area=area
            )
            
            # Label: 1 if match (similar domain/tech), 0 otherwise
            match = 1 if np.random.random() > 0.3 else 0
            if not match:
                # Shuffle to make it different
                random_domain = np.random.choice(domains)
                job = job.replace(domain, random_domain)
            
            resumes.append(resume)
            jobs.append(job)
            labels.append(match)
        
        return resumes, jobs, labels

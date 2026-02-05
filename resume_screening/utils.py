"""
Utility functions for resume screening system
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


def save_results(results: List[Dict], output_file: str):
    """
    Save ranking results to JSON
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def load_results(input_file: str) -> List[Dict]:
    """
    Load ranking results from JSON
    
    Args:
        input_file: Input file path
        
    Returns:
        List of result dictionaries
    """
    with open(input_file, 'r') as f:
        return json.load(f)


def format_ranking_results(resume_indices: List[int], 
                          resume_names: List[str],
                          scores: List[float]) -> str:
    """
    Format ranking results for display
    
    Args:
        resume_indices: Indices of resumes
        resume_names: Names of resumes
        scores: Similarity scores
        
    Returns:
        Formatted string
    """
    output = "Resume Rankings\n"
    output += "=" * 50 + "\n"
    
    for rank, (idx, name, score) in enumerate(zip(resume_indices, resume_names, scores), 1):
        output += f"{rank}. {name}\n"
        output += f"   Score: {score:.4f}\n"
        output += f"   Match: {score*100:.2f}%\n"
        output += "\n"
    
    return output


def create_report(rankings: List[Dict], output_file: str = None) -> str:
    """
    Create comprehensive ranking report
    
    Args:
        rankings: List of ranking dictionaries
        output_file: Optional output file path
        
    Returns:
        Report string
    """
    report = f"Resume Screening Report\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "=" * 70 + "\n\n"
    
    report += f"Total Resumes Processed: {len(rankings)}\n\n"
    
    # Top recommendations
    top_n = min(5, len(rankings))
    report += f"Top {top_n} Recommendations:\n"
    report += "-" * 70 + "\n"
    
    for i, ranking in enumerate(rankings[:top_n], 1):
        report += f"{i}. Resume: {ranking.get('name', f'Resume {i}')}\n"
        report += f"   Overall Score: {ranking.get('score', 0):.4f}\n"
        
        if 'details' in ranking:
            for metric, value in ranking['details'].items():
                report += f"   {metric}: {value:.4f}\n"
        
        report += "\n"
    
    # Statistics
    scores = [r.get('score', 0) for r in rankings]
    report += f"\nStatistics:\n"
    report += f"Average Score: {sum(scores)/len(scores):.4f}\n"
    report += f"Max Score: {max(scores):.4f}\n"
    report += f"Min Score: {min(scores):.4f}\n"
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report


def validate_input_texts(resumes: List[str], job_description: str) -> bool:
    """
    Validate input texts
    
    Args:
        resumes: List of resume texts
        job_description: Job description text
        
    Returns:
        True if valid, False otherwise
    """
    logger = get_logger(__name__)
    
    if not resumes:
        logger.error("No resumes provided")
        return False
    
    if not job_description:
        logger.error("No job description provided")
        return False
    
    if not isinstance(resumes, list):
        logger.error("Resumes must be a list")
        return False
    
    if not isinstance(job_description, str):
        logger.error("Job description must be a string")
        return False
    
    return True


def batch_process(items: List[Any], batch_size: int = 32):
    """
    Create batches from list
    
    Args:
        items: List of items
        batch_size: Batch size
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1]
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.times = {}
    
    def start(self, task_name: str):
        """Start timing a task"""
        import time
        self.times[task_name] = {'start': time.time()}
    
    def end(self, task_name: str) -> float:
        """End timing a task, return elapsed time"""
        import time
        if task_name in self.times:
            elapsed = time.time() - self.times[task_name]['start']
            self.times[task_name]['elapsed'] = elapsed
            return elapsed
        return 0
    
    def report(self) -> str:
        """Generate performance report"""
        report = "Performance Report\n"
        report += "=" * 40 + "\n"
        
        for task_name, data in self.times.items():
            if 'elapsed' in data:
                report += f"{task_name}: {data['elapsed']:.4f}s\n"
        
        total = sum(data.get('elapsed', 0) for data in self.times.values())
        report += f"Total: {total:.4f}s\n"
        
        return report

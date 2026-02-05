/**
 * Resume Screening System - Frontend JavaScript
 */

// Global state
let resumeCount = 1;
let analysisResults = null;

// Sample data for testing
const sampleJobDescription = `Senior Python Developer - Machine Learning

We are looking for an experienced Python developer with strong ML background.

Requirements:
- 5+ years Python development experience
- Strong machine learning background (TensorFlow/PyTorch)
- Django or FastAPI experience
- AWS cloud expertise
- Experience building production ML systems
- Knowledge of NLP techniques
- Git version control

Responsibilities:
- Design and implement ML models
- Build Python APIs and services
- Optimize model performance
- Collaborate with data science team
- Code reviews and mentoring`;

const sampleResumes = [
    `John Smith
Senior Python Developer
Email: john.smith@email.com
Phone: (555) 123-4567

Summary:
Experienced Python developer with 7 years of experience in machine learning and web development.

Experience:
- Lead Python Developer at TechCorp (2019-Present)
  - Built ML pipelines using TensorFlow and PyTorch
  - Developed Django REST APIs serving 1M+ requests/day
  - Implemented AWS infrastructure with EC2, S3, Lambda

Skills:
Python, Django, FastAPI, TensorFlow, PyTorch, AWS, Docker, Kubernetes, Git, PostgreSQL, Redis, Machine Learning, NLP`,

    `Jane Doe
Java Backend Engineer
Email: jane.doe@email.com
Phone: (555) 987-6543

Summary:
5 years of experience in Java enterprise development.

Experience:
- Senior Java Developer at FinTech Inc (2020-Present)
  - Built microservices using Spring Boot
  - Kubernetes orchestration and deployment
  - PostgreSQL database optimization

Skills:
Java, Spring Boot, Kubernetes, PostgreSQL, Kafka, Docker, Maven, Jenkins, AWS, Microservices`,

    `Bob Wilson
Full Stack Developer
Email: bob.wilson@email.com
Phone: (555) 456-7890

Summary:
6 years of web development experience with modern frameworks.

Experience:
- Full Stack Developer at WebAgency (2018-Present)
  - React and Node.js applications
  - MongoDB database design
  - Team leadership for 5 developers

Skills:
JavaScript, React, Node.js, MongoDB, CSS, HTML, Express, Redux, GraphQL, Git`
];

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('Resume Screening System initialized');
});

/**
 * Add a new resume input field
 */
function addResumeField() {
    resumeCount++;
    const container = document.getElementById('resumeContainer');
    
    const resumeDiv = document.createElement('div');
    resumeDiv.className = 'resume-input';
    resumeDiv.dataset.index = resumeCount - 1;
    resumeDiv.innerHTML = `
        <div class="resume-header">
            <span class="resume-label">Resume #${resumeCount}</span>
            <button class="btn-icon" onclick="removeResume(this)" title="Remove">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <textarea placeholder="Paste resume text here..."></textarea>
    `;
    
    container.appendChild(resumeDiv);
    resumeDiv.querySelector('textarea').focus();
}

/**
 * Remove a resume input field
 */
function removeResume(button) {
    const resumeInputs = document.querySelectorAll('.resume-input');
    if (resumeInputs.length > 1) {
        button.closest('.resume-input').remove();
        updateResumeLabels();
    } else {
        alert('You need at least one resume to analyze.');
    }
}

/**
 * Update resume labels after removal
 */
function updateResumeLabels() {
    const resumeInputs = document.querySelectorAll('.resume-input');
    resumeInputs.forEach((input, index) => {
        input.dataset.index = index;
        input.querySelector('.resume-label').textContent = `Resume #${index + 1}`;
    });
    resumeCount = resumeInputs.length;
}

/**
 * Load sample data for testing
 */
function loadSampleData() {
    // Set job description
    document.getElementById('jobDescription').value = sampleJobDescription;
    
    // Clear existing resumes
    const container = document.getElementById('resumeContainer');
    container.innerHTML = '';
    resumeCount = 0;
    
    // Add sample resumes
    sampleResumes.forEach((resume, index) => {
        resumeCount++;
        const resumeDiv = document.createElement('div');
        resumeDiv.className = 'resume-input';
        resumeDiv.dataset.index = index;
        resumeDiv.innerHTML = `
            <div class="resume-header">
                <span class="resume-label">Resume #${index + 1}</span>
                <button class="btn-icon" onclick="removeResume(this)" title="Remove">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <textarea placeholder="Paste resume text here...">${resume}</textarea>
        `;
        container.appendChild(resumeDiv);
    });
    
    // Smooth scroll to screening section
    document.getElementById('screen').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Analyze resumes using the backend API
 */
async function analyzeResumes() {
    const jobDescription = document.getElementById('jobDescription').value.trim();
    const resumeTextareas = document.querySelectorAll('.resume-input textarea');
    
    // Validation
    if (!jobDescription) {
        alert('Please enter a job description.');
        document.getElementById('jobDescription').focus();
        return;
    }
    
    const resumes = [];
    resumeTextareas.forEach(textarea => {
        const text = textarea.value.trim();
        if (text) {
            resumes.push(text);
        }
    });
    
    if (resumes.length === 0) {
        alert('Please add at least one resume.');
        return;
    }
    
    // Show loading
    showLoading(true);
    hideResults();
    
    try {
        const response = await fetch('/api/rank', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_description: jobDescription,
                resumes: resumes
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            analysisResults = data;
            displayResults(data);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Error analyzing resumes: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    const rankings = data.rankings;
    const details = data.details || {};
    
    rankings.forEach((item, index) => {
        const rank = index + 1;
        const resumeIdx = item.resume_index;
        const score = item.score;
        const resumeDetails = details[resumeIdx] || {};
        
        const card = document.createElement('div');
        card.className = `result-card rank-${rank}`;
        
        // Extract name from resume (first line typically)
        const resumeText = document.querySelectorAll('.resume-input textarea')[resumeIdx]?.value || '';
        const firstLine = resumeText.split('\n')[0].trim();
        const candidateName = firstLine || `Candidate ${resumeIdx + 1}`;
        
        // Get skills
        const skills = resumeDetails.skills || extractSkillsFromText(resumeText);
        const emails = resumeDetails.emails || extractEmails(resumeText);
        const phones = resumeDetails.phones || extractPhones(resumeText);
        
        card.innerHTML = `
            <div class="result-header">
                <div class="rank-badge">${rank}</div>
                <div class="result-info">
                    <h4>${candidateName}</h4>
                    <p>Resume #${resumeIdx + 1}</p>
                </div>
                <div class="match-score">
                    <span class="score-value">${(score * 100).toFixed(1)}%</span>
                    <span class="score-label">Match Score</span>
                    <div class="score-bar">
                        <div class="score-bar-fill" style="width: ${score * 100}%"></div>
                    </div>
                </div>
            </div>
            <div class="result-body">
                <div class="skills-section">
                    <h5>Extracted Skills</h5>
                    <div class="skills-tags">
                        ${skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                    </div>
                </div>
                ${(emails.length || phones.length) ? `
                <div class="contact-info">
                    ${emails.length ? `<span><i class="fas fa-envelope"></i> ${emails[0]}</span>` : ''}
                    ${phones.length ? `<span><i class="fas fa-phone"></i> ${phones[0]}</span>` : ''}
                </div>
                ` : ''}
            </div>
        `;
        
        container.appendChild(card);
    });
    
    showResults();
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Extract skills from text (client-side fallback)
 */
function extractSkillsFromText(text) {
    const skillPatterns = [
        'python', 'java', 'javascript', 'typescript', 'c\\+\\+', 'c#', 'go', 'rust', 'ruby', 'php',
        'react', 'angular', 'vue', 'node\\.js', 'django', 'flask', 'fastapi', 'spring',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
        'git', 'jenkins', 'ci/cd', 'agile', 'scrum',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science',
        'rest api', 'graphql', 'microservices'
    ];
    
    const foundSkills = new Set();
    const lowerText = text.toLowerCase();
    
    skillPatterns.forEach(pattern => {
        const regex = new RegExp(`\\b${pattern}\\b`, 'i');
        if (regex.test(lowerText)) {
            // Capitalize properly
            const skill = pattern.replace(/\\./g, '.').replace(/\\\+/g, '+');
            foundSkills.add(skill.charAt(0).toUpperCase() + skill.slice(1));
        }
    });
    
    return Array.from(foundSkills).slice(0, 10);
}

/**
 * Extract emails from text
 */
function extractEmails(text) {
    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
    return text.match(emailRegex) || [];
}

/**
 * Extract phone numbers from text
 */
function extractPhones(text) {
    const phoneRegex = /[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}/g;
    return text.match(phoneRegex) || [];
}

/**
 * Export results to JSON
 */
function exportResults() {
    if (!analysisResults) {
        alert('No results to export. Please analyze resumes first.');
        return;
    }
    
    const exportData = {
        timestamp: new Date().toISOString(),
        job_description: document.getElementById('jobDescription').value,
        rankings: analysisResults.rankings.map((item, index) => ({
            rank: index + 1,
            resume_index: item.resume_index,
            score: item.score,
            score_percentage: (item.score * 100).toFixed(2) + '%'
        }))
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `resume-screening-results-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Show/hide loading indicator
 */
function showLoading(show) {
    const loading = document.getElementById('loadingIndicator');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (show) {
        loading.classList.remove('hidden');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    } else {
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze & Rank Resumes';
    }
}

/**
 * Show results section
 */
function showResults() {
    document.getElementById('resultsSection').classList.remove('hidden');
}

/**
 * Hide results section
 */
function hideResults() {
    document.getElementById('resultsSection').classList.add('hidden');
}

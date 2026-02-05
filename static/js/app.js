/**
 * Resume Screening System - Frontend JavaScript
 * Supports PDF, DOCX, URL, and text input
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
    console.log('Resume Screening System v2.0 initialized');
    console.log('Supported formats: PDF, DOCX, DOC, TXT, URL');
});

/**
 * Switch between file, url, and text input modes
 */
function switchInputMode(button, mode) {
    const resumeInput = button.closest('.resume-input');
    const tabs = resumeInput.querySelectorAll('.upload-tab');
    const modes = resumeInput.querySelectorAll('.upload-mode');
    
    // Update tabs
    tabs.forEach(tab => tab.classList.remove('active'));
    button.classList.add('active');
    
    // Update modes
    modes.forEach(m => m.classList.remove('active'));
    resumeInput.querySelector(`.${mode}-mode`).classList.add('active');
}

/**
 * Handle file drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.add('drag-over');
}

/**
 * Handle file drag leave
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('drag-over');
}

/**
 * Handle file drop
 */
function handleFileDrop(event, dropZone) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0], dropZone);
    }
}

/**
 * Handle file selection via input
 */
function handleFileSelect(event, input) {
    const file = event.target.files[0];
    if (file) {
        const dropZone = input.closest('.file-drop-zone');
        processFile(file, dropZone);
    }
}

/**
 * Process uploaded file
 */
async function processFile(file, dropZone) {
    const resumeInput = dropZone.closest('.resume-input');
    const dropContent = dropZone.querySelector('.drop-zone-content');
    const fileInfo = dropZone.querySelector('.file-info');
    const fileName = fileInfo.querySelector('.file-name');
    const fileStatus = fileInfo.querySelector('.file-status');
    
    // Check file type
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword', 'text/plain'];
    const allowedExtensions = ['.pdf', '.docx', '.doc', '.txt'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(ext)) {
        alert('Please upload a PDF, DOCX, DOC, or TXT file.');
        return;
    }
    
    // Show loading state
    dropContent.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    fileName.textContent = file.name;
    fileStatus.textContent = 'Extracting text...';
    fileStatus.className = 'file-status loading';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            fileStatus.textContent = '✓ Loaded';
            fileStatus.className = 'file-status success';
            
            // Store the extracted text
            const storage = resumeInput.querySelector('.resume-text-storage');
            storage.value = data.text;
            
            // Show preview
            showResumePreview(resumeInput, data);
        } else {
            throw new Error(data.error || 'Failed to process file');
        }
    } catch (error) {
        console.error('File upload error:', error);
        fileStatus.textContent = '✗ Error';
        fileStatus.className = 'file-status error';
        alert(`Error processing file: ${error.message}`);
        
        // Reset drop zone
        setTimeout(() => {
            dropContent.classList.remove('hidden');
            fileInfo.classList.add('hidden');
        }, 2000);
    }
}

/**
 * Fetch resume from URL
 */
async function fetchFromUrl(button) {
    const resumeInput = button.closest('.resume-input');
    const urlInput = resumeInput.querySelector('.url-input');
    const url = urlInput.value.trim();
    
    if (!url) {
        alert('Please enter a URL.');
        urlInput.focus();
        return;
    }
    
    // Show loading state
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    button.disabled = true;
    
    try {
        const response = await fetch('/api/fetch-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Store the extracted text
            const storage = resumeInput.querySelector('.resume-text-storage');
            storage.value = data.text;
            
            // Show preview
            showResumePreview(resumeInput, data);
            
            button.innerHTML = '<i class="fas fa-check"></i> Loaded';
            button.className = 'btn btn-small btn-success';
        } else {
            throw new Error(data.error || 'Failed to fetch URL');
        }
    } catch (error) {
        console.error('URL fetch error:', error);
        alert(`Error fetching URL: ${error.message}`);
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

/**
 * Show resume preview with extracted info
 */
function showResumePreview(resumeInput, data) {
    const preview = resumeInput.querySelector('.resume-preview');
    const uploadOptions = resumeInput.querySelector('.upload-options');
    
    // Update stats
    preview.querySelector('.word-count').textContent = data.preview?.word_count || data.text.split(/\s+/).length;
    preview.querySelector('.skill-count').textContent = data.preview?.skills?.length || 0;
    
    // Show skills
    const skillsContainer = preview.querySelector('.preview-skills');
    const skills = data.preview?.skills || [];
    skillsContainer.innerHTML = skills.slice(0, 8).map(skill => 
        `<span class="skill-tag">${skill}</span>`
    ).join('');
    
    // Show preview, hide upload options
    uploadOptions.classList.add('hidden');
    preview.classList.remove('hidden');
}

/**
 * Clear resume and reset to upload mode
 */
function clearResume(button) {
    const resumeInput = button.closest('.resume-input');
    const preview = resumeInput.querySelector('.resume-preview');
    const uploadOptions = resumeInput.querySelector('.upload-options');
    const storage = resumeInput.querySelector('.resume-text-storage');
    const dropZone = resumeInput.querySelector('.file-drop-zone');
    const textArea = resumeInput.querySelector('.text-mode textarea');
    const urlInput = resumeInput.querySelector('.url-input');
    
    // Clear all inputs
    storage.value = '';
    if (textArea) textArea.value = '';
    if (urlInput) urlInput.value = '';
    
    // Reset drop zone
    if (dropZone) {
        dropZone.querySelector('.drop-zone-content').classList.remove('hidden');
        dropZone.querySelector('.file-info').classList.add('hidden');
        const fileInput = dropZone.querySelector('.file-input');
        if (fileInput) fileInput.value = '';
    }
    
    // Reset URL button
    const urlButton = resumeInput.querySelector('.url-mode .btn');
    if (urlButton) {
        urlButton.innerHTML = '<i class="fas fa-download"></i> Fetch';
        urlButton.className = 'btn btn-small btn-primary';
        urlButton.disabled = false;
    }
    
    // Show upload options, hide preview
    uploadOptions.classList.remove('hidden');
    preview.classList.add('hidden');
}

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
        <div class="upload-options">
            <div class="upload-tabs">
                <button class="upload-tab active" onclick="switchInputMode(this, 'file')" data-mode="file">
                    <i class="fas fa-upload"></i> Upload File
                </button>
                <button class="upload-tab" onclick="switchInputMode(this, 'url')" data-mode="url">
                    <i class="fas fa-link"></i> URL
                </button>
                <button class="upload-tab" onclick="switchInputMode(this, 'text')" data-mode="text">
                    <i class="fas fa-keyboard"></i> Paste Text
                </button>
            </div>
            <div class="upload-content">
                <!-- File Upload -->
                <div class="upload-mode file-mode active">
                    <div class="file-drop-zone" ondrop="handleFileDrop(event, this)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                        <input type="file" class="file-input" accept=".pdf,.docx,.doc,.txt" onchange="handleFileSelect(event, this)">
                        <div class="drop-zone-content">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <p>Drop PDF, DOCX, or DOC file here</p>
                            <span>or click to browse</span>
                        </div>
                        <div class="file-info hidden">
                            <i class="fas fa-file-alt"></i>
                            <span class="file-name"></span>
                            <span class="file-status"></span>
                        </div>
                    </div>
                </div>
                <!-- URL Input -->
                <div class="upload-mode url-mode">
                    <div class="url-input-wrapper">
                        <input type="url" class="url-input" placeholder="https://example.com/resume.pdf">
                        <button class="btn btn-small btn-primary" onclick="fetchFromUrl(this)">
                            <i class="fas fa-download"></i> Fetch
                        </button>
                    </div>
                    <p class="url-hint">Supports PDF, DOCX links or any webpage with resume content</p>
                </div>
                <!-- Text Input -->
                <div class="upload-mode text-mode">
                    <textarea placeholder="Paste resume text here..."></textarea>
                </div>
            </div>
        </div>
        <div class="resume-preview hidden">
            <div class="preview-header">
                <span><i class="fas fa-check-circle"></i> Resume Loaded</span>
                <button class="btn-icon" onclick="clearResume(this)" title="Clear">
                    <i class="fas fa-redo"></i>
                </button>
            </div>
            <div class="preview-stats">
                <span class="stat-item"><i class="fas fa-file-word"></i> <span class="word-count">0</span> words</span>
                <span class="stat-item"><i class="fas fa-code"></i> <span class="skill-count">0</span> skills detected</span>
            </div>
            <div class="preview-skills"></div>
        </div>
        <textarea class="resume-text-storage hidden"></textarea>
    `;
    
    container.appendChild(resumeDiv);
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
    
    // Add sample resumes with text mode
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
            <div class="upload-options hidden">
                <div class="upload-tabs">
                    <button class="upload-tab" onclick="switchInputMode(this, 'file')" data-mode="file">
                        <i class="fas fa-upload"></i> Upload File
                    </button>
                    <button class="upload-tab" onclick="switchInputMode(this, 'url')" data-mode="url">
                        <i class="fas fa-link"></i> URL
                    </button>
                    <button class="upload-tab active" onclick="switchInputMode(this, 'text')" data-mode="text">
                        <i class="fas fa-keyboard"></i> Paste Text
                    </button>
                </div>
                <div class="upload-content">
                    <div class="upload-mode file-mode">
                        <div class="file-drop-zone" ondrop="handleFileDrop(event, this)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                            <input type="file" class="file-input" accept=".pdf,.docx,.doc,.txt" onchange="handleFileSelect(event, this)">
                            <div class="drop-zone-content">
                                <i class="fas fa-cloud-upload-alt"></i>
                                <p>Drop PDF, DOCX, or DOC file here</p>
                                <span>or click to browse</span>
                            </div>
                            <div class="file-info hidden">
                                <i class="fas fa-file-alt"></i>
                                <span class="file-name"></span>
                                <span class="file-status"></span>
                            </div>
                        </div>
                    </div>
                    <div class="upload-mode url-mode">
                        <div class="url-input-wrapper">
                            <input type="url" class="url-input" placeholder="https://example.com/resume.pdf">
                            <button class="btn btn-small btn-primary" onclick="fetchFromUrl(this)">
                                <i class="fas fa-download"></i> Fetch
                            </button>
                        </div>
                        <p class="url-hint">Supports PDF, DOCX links or any webpage with resume content</p>
                    </div>
                    <div class="upload-mode text-mode active">
                        <textarea placeholder="Paste resume text here..."></textarea>
                    </div>
                </div>
            </div>
            <div class="resume-preview">
                <div class="preview-header">
                    <span><i class="fas fa-check-circle"></i> Sample Resume Loaded</span>
                    <button class="btn-icon" onclick="clearResume(this)" title="Clear">
                        <i class="fas fa-redo"></i>
                    </button>
                </div>
                <div class="preview-stats">
                    <span class="stat-item"><i class="fas fa-file-word"></i> <span class="word-count">${resume.split(/\s+/).length}</span> words</span>
                    <span class="stat-item"><i class="fas fa-code"></i> <span class="skill-count">${extractSkillsFromText(resume).length}</span> skills detected</span>
                </div>
                <div class="preview-skills">
                    ${extractSkillsFromText(resume).slice(0, 6).map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
            <textarea class="resume-text-storage hidden">${resume}</textarea>
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
    const resumeInputs = document.querySelectorAll('.resume-input');
    
    // Validation
    if (!jobDescription) {
        alert('Please enter a job description.');
        document.getElementById('jobDescription').focus();
        return;
    }
    
    // Collect resume texts from storage (for uploaded files) or text mode textarea
    const resumes = [];
    resumeInputs.forEach((input, idx) => {
        // First check the hidden storage (for uploaded files/URLs)
        const storage = input.querySelector('.resume-text-storage');
        let text = storage?.value?.trim() || '';
        
        // If no stored text, check the text mode textarea
        if (!text) {
            const textArea = input.querySelector('.text-mode textarea');
            text = textArea?.value?.trim() || '';
        }
        
        if (text) {
            resumes.push(text);
        }
    });
    
    if (resumes.length === 0) {
        alert('Please add at least one resume (upload a file, enter a URL, or paste text).');
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
        
        // Get resume text from storage (for uploaded files) or text mode textarea
        const resumeInputs = document.querySelectorAll('.resume-input');
        let resumeText = '';
        if (resumeInputs[resumeIdx]) {
            const storage = resumeInputs[resumeIdx].querySelector('.resume-text-storage');
            resumeText = storage?.value || '';
            if (!resumeText) {
                const textArea = resumeInputs[resumeIdx].querySelector('.text-mode textarea');
                resumeText = textArea?.value || '';
            }
        }
        
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

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileText, Target, Zap, Download, Save, CheckCircle, AlertCircle, Clock, Star } from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Component for uploading and managing resume templates
const ResumeUpload = ({ onResumeUpload, currentTemplate }) => {
  const [dragOver, setDragOver] = useState(false);
  const [uploadMethod, setUploadMethod] = useState('text');

  const handleFileUpload = async (file) => {
    if (file && file.type === 'text/plain') {
      const text = await file.text();
      onResumeUpload(text, file.name);
    } else {
      alert('Currently supporting text files only. PDF/DOCX support coming soon!');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleTextSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const resumeText = formData.get('resumeText');
    if (resumeText.trim().length > 50) {
      onResumeUpload(resumeText, 'pasted-resume.txt');
    } else {
      alert('Please enter at least 50 characters of resume content');
    }
  };

  return (
    <div className="resume-upload-container">
      <div className="upload-method-tabs">
        <button 
          className={`tab-button ${uploadMethod === 'file' ? 'active' : ''}`}
          onClick={() => setUploadMethod('file')}
          data-testid="file-upload-tab"
        >
          <Upload className="icon" />
          Upload File
        </button>
        <button 
          className={`tab-button ${uploadMethod === 'text' ? 'active' : ''}`}
          onClick={() => setUploadMethod('text')}
          data-testid="text-input-tab"
        >
          <FileText className="icon" />
          Paste Text
        </button>
      </div>

      {uploadMethod === 'file' ? (
        <div 
          className={`file-drop-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          data-testid="file-drop-zone"
        >
          <div className="drop-content">
            <Upload className="upload-icon" />
            <h3>Drop your resume here</h3>
            <p>or click to browse files</p>
            <input
              type="file"
              accept=".txt,.pdf,.docx"
              onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
              className="hidden-file-input"
              data-testid="file-input"
            />
            <div className="supported-formats">
              Supported: TXT, PDF, DOCX (TXT only for now)
            </div>
          </div>
        </div>
      ) : (
        <form onSubmit={handleTextSubmit} className="text-input-form">
          <textarea
            name="resumeText"
            placeholder="Paste your resume content here..."
            className="resume-textarea"
            rows="12"
            required
            minLength="50"
            data-testid="resume-textarea"
          />
          <button type="submit" className="primary-button" data-testid="submit-resume-text">
            <Save className="icon" />
            Set as My Template
          </button>
        </form>
      )}

      {currentTemplate && (
        <div className="current-template-info" data-testid="current-template">
          <CheckCircle className="icon success" />
          <div>
            <strong>Current Template:</strong> {currentTemplate.filename}
            <div className="template-stats">
              {currentTemplate.wordCount} words • Uploaded {new Date(currentTemplate.uploadedAt).toLocaleDateString()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Component for job description input
const JobDescriptionInput = ({ onJobSubmit, isAnalyzing }) => {
  const [jobData, setJobData] = useState({
    description: '',
    title: '',
    company: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (jobData.description.trim().length > 30) {
      onJobSubmit(jobData);
    } else {
      alert('Please enter at least 30 characters of job description');
    }
  };

  return (
    <div className="job-input-container">
      <div className="section-header">
        <Target className="icon" />
        <h2>Target Job (Optional)</h2>
        <span className="subtitle">Add a job description to get targeted optimization</span>
      </div>

      <form onSubmit={handleSubmit} className="job-form">
        <div className="form-row">
          <input
            type="text"
            placeholder="Job Title (e.g., Senior Developer)"
            value={jobData.title}
            onChange={(e) => setJobData({...jobData, title: e.target.value})}
            className="job-input"
            data-testid="job-title-input"
          />
          <input
            type="text"
            placeholder="Company Name"
            value={jobData.company}
            onChange={(e) => setJobData({...jobData, company: e.target.value})}
            className="job-input"
            data-testid="company-input"
          />
        </div>
        
        <textarea
          placeholder="Paste the job description here to get targeted suggestions..."
          value={jobData.description}
          onChange={(e) => setJobData({...jobData, description: e.target.value})}
          className="job-textarea"
          rows="6"
          data-testid="job-description-textarea"
        />
        
        <button 
          type="submit" 
          className="analyze-button"
          disabled={isAnalyzing}
          data-testid="analyze-button"
        >
          <Zap className="icon" />
          {isAnalyzing ? 'Analyzing...' : 'Analyze & Optimize'}
        </button>
      </form>
    </div>
  );
};

// Component to display analysis results with playground
const AnalysisResults = ({ results, onOptimizeSection, currentTemplate, onGenerateOptimized }) => {
  const [selectedSuggestion, setSelectedSuggestion] = useState(null);
  const [optimizedResume, setOptimizedResume] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showPlayground, setShowPlayground] = useState(false);

  if (!results) return null;

  const formatScore = (score) => Math.round(score * 100);

  const handleGenerateOptimized = async () => {
    setIsGenerating(true);
    try {
      const formData = new FormData();
      formData.append('resume_text', currentTemplate.content);
      
      // Add job context if available
      const jobData = sessionStorage.getItem('lastJobData');
      if (jobData) {
        const job = JSON.parse(jobData);
        if (job.description) formData.append('job_description', job.description);
        if (job.title) formData.append('job_title', job.title);
        if (job.company) formData.append('company', job.company);
      }

      const response = await axios.post(`${API}/generate-optimized-resume`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setOptimizedResume(response.data);
      setShowPlayground(true);
    } catch (error) {
      console.error('Optimization failed:', error);
      alert('Failed to generate optimized resume. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadResume = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="analysis-results" data-testid="analysis-results">
      <div className="results-header">
        <CheckCircle className="icon success" />
        <h2>Analysis Complete</h2>
        <div className="processing-time">
          <Clock className="icon" />
          Processed in {results.processing_time?.toFixed(1)}s
        </div>
      </div>

      {/* Score Summary */}
      <div className="score-summary">
        <div className="score-card">
          <div className="score-value">{Math.round(results.ats_score)}</div>
          <div className="score-label">ATS Score</div>
        </div>
        {results.job_match_score !== null && (
          <div className="score-card">
            <div className="score-value">{formatScore(results.job_match_score)}%</div>
            <div className="score-label">Job Match</div>
          </div>
        )}
        <div className="score-card">
          <div className="score-value">{results.skill_analysis?.identified_skills?.length || 0}</div>
          <div className="score-label">Skills Found</div>
        </div>
      </div>

      {/* Resume Summary */}
      <div className="summary-section">
        <h3>Resume Summary</h3>
        <p className="summary-text" data-testid="resume-summary">
          {results.resume_summary}
        </p>
      </div>

      {/* Skills Analysis */}
      {results.skill_analysis && (
        <div className="skills-section">
          <h3>Skills Analysis</h3>
          <div className="skills-grid">
            {results.skill_analysis.identified_skills?.length > 0 && (
              <div className="skill-category">
                <h4>✅ Skills Found</h4>
                <div className="skill-tags">
                  {results.skill_analysis.identified_skills.map((skill, index) => (
                    <span key={index} className="skill-tag found" data-testid={`found-skill-${index}`}>
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {results.skill_analysis.missing_skills?.length > 0 && (
              <div className="skill-category">
                <h4>⚠️ Missing Keywords</h4>
                <div className="skill-tags">
                  {results.skill_analysis.missing_skills.map((skill, index) => (
                    <span key={index} className="skill-tag missing" data-testid={`missing-skill-${index}`}>
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Optimization Suggestions */}
      {results.optimization_suggestions?.length > 0 && (
        <div className="suggestions-section">
          <h3>Optimization Suggestions</h3>
          <div className="suggestions-list">
            {results.optimization_suggestions.map((suggestion, index) => (
              <div 
                key={index} 
                className={`suggestion-card priority-${suggestion.priority}`}
                data-testid={`suggestion-${index}`}
              >
                <div className="suggestion-header">
                  <span className="section-name">{suggestion.section}</span>
                  <span className={`priority-badge ${suggestion.priority}`}>
                    {suggestion.priority} priority
                  </span>
                </div>
                
                <div className="suggestion-content">
                  <div className="suggestion-text">
                    <strong>Suggestion:</strong> {suggestion.reason}
                  </div>
                  
                  {suggestion.original_text && (
                    <div className="text-comparison">
                      <div className="original-text">
                        <label>Current:</label>
                        <p>{suggestion.original_text}</p>
                      </div>
                      <div className="optimized-text">
                        <label>Suggested:</label>
                        <p>{suggestion.optimized_text}</p>
                      </div>
                    </div>
                  )}
                </div>
                
                <button 
                  className="apply-suggestion-btn"
                  onClick={() => setSelectedSuggestion(suggestion)}
                  data-testid={`apply-suggestion-${index}`}
                >
                  Apply Suggestion
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Main App Component
const App = () => {
  const [currentTemplate, setCurrentTemplate] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  // Load template from localStorage on startup
  useEffect(() => {
    const savedTemplate = localStorage.getItem('resumeTemplate');
    if (savedTemplate) {
      try {
        setCurrentTemplate(JSON.parse(savedTemplate));
      } catch (e) {
        console.error('Error loading saved template:', e);
      }
    }
  }, []);

  const handleResumeUpload = (resumeText, filename) => {
    const template = {
      content: resumeText,
      filename: filename,
      uploadedAt: new Date().toISOString(),
      wordCount: resumeText.split(' ').length,
      templateId: `template_${Date.now()}`
    };
    
    setCurrentTemplate(template);
    localStorage.setItem('resumeTemplate', JSON.stringify(template));
    setAnalysisResults(null); // Clear previous results
    setError(null);
  };

  const handleJobSubmit = async (jobData) => {
    if (!currentTemplate) {
      setError('Please upload a resume first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('resume_text', currentTemplate.content);
      
      if (jobData.description.trim()) {
        formData.append('job_description', jobData.description);
      }
      if (jobData.title.trim()) {
        formData.append('job_title', jobData.title);
      }
      if (jobData.company.trim()) {
        formData.append('company', jobData.company);
      }
      formData.append('template_id', currentTemplate.templateId);

      const response = await axios.post(`${API}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setAnalysisResults(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      setError(
        error.response?.data?.detail || 
        'Analysis failed. Please ensure Ollama is installed and running.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleOptimizeSection = async (sectionText, sectionType) => {
    // Future implementation for section-specific optimization
    console.log('Optimizing section:', sectionType, sectionText);
  };

  const clearTemplate = () => {
    setCurrentTemplate(null);
    setAnalysisResults(null);
    localStorage.removeItem('resumeTemplate');
  };

  return (
    <div className="App" data-testid="main-app">
      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="header-content">
            <div className="logo-section">
              <div className="logo-icon">
                <FileText />
              </div>
              <div className="logo-text">
                <h1>AI Resume Optimizer</h1>
                <p>Template-based resume optimization with free AI</p>
              </div>
            </div>
            
            {currentTemplate && (
              <button 
                onClick={clearTemplate} 
                className="clear-template-btn"
                data-testid="clear-template-button"
              >
                Clear Template
              </button>
            )}
          </div>
        </header>

        {/* Error Display */}
        {error && (
          <div className="error-banner" data-testid="error-banner">
            <AlertCircle className="icon" />
            <span>{error}</span>
            <button onClick={() => setError(null)} className="close-error">×</button>
          </div>
        )}

        {/* Main Content */}
        <main className="main-content">
          {!currentTemplate ? (
            /* Resume Upload Step */
            <div className="upload-step">
              <div className="step-header">
                <Star className="icon" />
                <h2>Step 1: Upload Your Resume Template</h2>
                <p>This becomes your default template for all optimizations</p>
              </div>
              <ResumeUpload 
                onResumeUpload={handleResumeUpload}
                currentTemplate={currentTemplate}
              />
            </div>
          ) : (
            /* Analysis Step */
            <div className="analysis-step">
              <JobDescriptionInput 
                onJobSubmit={handleJobSubmit}
                isAnalyzing={isAnalyzing}
              />
              
              {isAnalyzing && (
                <div className="analyzing-indicator" data-testid="analyzing-indicator">
                  <div className="spinner"></div>
                  <p>Analyzing your resume with AI...</p>
                  <small>This may take 30-60 seconds</small>
                </div>
              )}
              
              <AnalysisResults 
                results={analysisResults}
                onOptimizeSection={handleOptimizeSection}
              />
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="app-footer">
          <div className="footer-content">
            <p>Powered by Ollama (Free AI) • Template-based optimization • Privacy-first design</p>
            <div className="footer-links">
              <a href="#" onClick={(e) => { e.preventDefault(); alert('Feature coming soon!'); }}>
                Download Optimized Resume
              </a>
              <a href="#" onClick={(e) => { e.preventDefault(); alert('Help documentation coming soon!'); }}>
                Help & Guide
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio
import time
import json
import re

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (keeping original setup)
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="AI Resume Optimizer", version="1.0.0")

# Create API router
api_router = APIRouter(prefix="/api")

# Ollama integration for free AI processing
try:
    import ollama
    OLLAMA_AVAILABLE = True
    OLLAMA_MODEL = "llama3.2"  # Using latest free model
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available. Install with: pip install ollama")

# Pydantic Models
class ResumeUpload(BaseModel):
    content: str = Field(..., min_length=50, description="Resume content as text")
    filename: Optional[str] = Field(None, description="Original filename")
    template_id: Optional[str] = Field(None, description="Template identifier")
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 50:
            raise ValueError('Resume content too short (minimum 50 characters)')
        return v.strip()

class JobDescription(BaseModel):
    content: str = Field(..., min_length=30, description="Job description content")
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")

class SkillAnalysis(BaseModel):
    identified_skills: List[str]
    missing_skills: List[str]
    skill_match_score: float = Field(ge=0.0, le=1.0)
    skill_categories: Dict[str, List[str]] = {}

class OptimizationSuggestion(BaseModel):
    section: str
    original_text: str
    optimized_text: str
    reason: str
    priority: str = Field(pattern="^(high|medium|low)$")

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    resume_summary: str
    job_match_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    skill_analysis: SkillAnalysis
    optimization_suggestions: List[OptimizationSuggestion]
    processing_time: float
    ats_score: float = Field(ge=0.0, le=100.0)
    template_preserved: bool = True

class TemplateInfo(BaseModel):
    template_id: str
    original_content: str
    structure: Dict[str, Any]
    created_at: datetime

# Ollama Service Class
class OllamaResumeOptimizer:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name
        self.ensure_model_available()
    
    def ensure_model_available(self):
        """Ensure the Ollama model is available locally"""
        if not OLLAMA_AVAILABLE:
            raise Exception("Ollama not installed. Please install Ollama and the required model.")
        
        try:
            # Check if model is available
            available_models = ollama.list()
            model_names = [model['name'] for model in available_models.get('models', [])]
            
            if self.model_name not in model_names:
                print(f"Pulling {self.model_name} model... This may take a few minutes.")
                ollama.pull(self.model_name)
                print(f"Model {self.model_name} is ready!")
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
    
    async def analyze_resume(self, resume_content: str, job_description: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive resume analysis using Ollama with fallback"""
        
        # Try AI analysis first, but fallback quickly if it fails
        try:
            return await asyncio.wait_for(
                self._ai_analysis(resume_content, job_description), 
                timeout=20.0  # 20 second timeout
            )
        except asyncio.TimeoutError:
            print("AI analysis timed out, using fallback analysis")
            return self._get_smart_fallback_analysis(resume_content, job_description)
        except Exception as e:
            print(f"AI analysis failed: {e}, using fallback")
            return self._get_smart_fallback_analysis(resume_content, job_description)
    
    async def _ai_analysis(self, resume_content: str, job_description: Optional[str] = None) -> Dict[str, Any]:
        """AI-powered analysis with Ollama"""
        
        analysis_prompt = f"""
        Analyze this resume comprehensively and provide detailed feedback:

        RESUME CONTENT:
        {resume_content}

        {f"JOB DESCRIPTION FOR COMPARISON: {job_description}" if job_description else ""}

        Please provide a JSON response with the following structure:
        {{
            "resume_summary": "Brief 2-3 sentence summary of the candidate",
            "identified_skills": ["skill1", "skill2", "skill3"],
            "skill_categories": {{
                "technical": ["Python", "JavaScript"],
                "soft": ["Leadership", "Communication"],
                "industry": ["Healthcare", "Finance"]
            }},
            "ats_compatibility_score": 85,
            "strengths": ["strength1", "strength2"],
            "improvement_areas": ["area1", "area2"],
            "keyword_suggestions": ["keyword1", "keyword2"],
            "job_match_score": {0.75 if job_description else 0.0}
        }}

        Focus on:
        1. ATS (Applicant Tracking System) compatibility
        2. Keyword optimization
        3. Content quality and relevance
        4. Professional presentation
        {"5. Match with job requirements" if job_description else ""}
        
        Respond ONLY with valid JSON, no additional text.
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=analysis_prompt,
                options={
                    'temperature': 0.3,  # Lower for consistent analysis
                    'top_p': 0.9,
                    'num_predict': 400  # Reduced for faster processing
                }
            )
            
            # Extract JSON from response
            response_text = response.get('response', '{}')
            
            # Try to extract JSON from the response
            try:
                # Look for JSON content between braces
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    return json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON parsing error: {e}")
                # Return default analysis if JSON parsing fails
                return self._get_smart_fallback_analysis(resume_content, job_description)
                
        except Exception as e:
            print(f"Ollama analysis error: {e}")
            return self._get_smart_fallback_analysis(resume_content, job_description)
    
    async def generate_optimization_suggestions(self, resume_content: str, job_description: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate specific optimization suggestions with fallback"""
        
        # Try AI generation first with timeout
        try:
            return await asyncio.wait_for(
                self._ai_suggestions(resume_content, job_description),
                timeout=15.0  # 15 second timeout
            )
        except asyncio.TimeoutError:
            print("AI suggestions timed out, using rule-based suggestions")
            return self._get_smart_suggestions(resume_content, job_description)
        except Exception as e:
            print(f"AI suggestions failed: {e}, using fallback")
            return self._get_smart_suggestions(resume_content, job_description)
    
    async def _ai_suggestions(self, resume_content: str, job_description: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate specific optimization suggestions"""
        
        optimization_prompt = f"""
        Analyze this resume and provide specific optimization suggestions:

        RESUME:
        {resume_content}

        {f"TARGET JOB: {job_description}" if job_description else ""}

        Provide suggestions in JSON format:
        {{
            "suggestions": [
                {{
                    "section": "Professional Summary",
                    "original_text": "Current text from resume",
                    "optimized_text": "Improved version with better keywords",
                    "reason": "Explanation of why this change improves the resume",
                    "priority": "high"
                }}
            ]
        }}

        Focus on:
        1. Adding relevant keywords
        2. Quantifying achievements 
        3. Using action verbs
        4. ATS optimization
        5. Professional language
        {"6. Matching job requirements" if job_description else ""}

        Respond ONLY with valid JSON.
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=optimization_prompt,
                options={
                    'temperature': 0.4,
                    'top_p': 0.9,
                    'num_predict': 300  # Reduced for faster processing
                }
            )
            
            response_text = response.get('response', '{}')
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)
                return result.get('suggestions', [])
            
            return []
            
        except Exception as e:
            print(f"Optimization suggestion error: {e}")
            return []
    
    def _get_smart_fallback_analysis(self, resume_content: str, job_description: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced fallback analysis when AI processing fails"""
        import re
        
        # Enhanced skill extraction
        technical_skills = ['python', 'javascript', 'java', 'c++', 'sql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 
                          'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'mongodb', 'postgresql',
                          'git', 'jenkins', 'ci/cd', 'devops', 'machine learning', 'ai', 'data science', 'tensorflow',
                          'pytorch', 'pandas', 'numpy', 'rest api', 'graphql', 'microservices', 'linux', 'unix']
        
        soft_skills = ['leadership', 'communication', 'problem solving', 'team work', 'project management',
                      'agile', 'scrum', 'collaboration', 'mentoring', 'strategic thinking']
        
        # Find skills in resume
        resume_lower = resume_content.lower()
        found_technical = [skill for skill in technical_skills if skill in resume_lower]
        found_soft = [skill for skill in soft_skills if skill in resume_lower]
        
        # Extract experience level
        years_pattern = r'(\d+)\+?\s*years?'
        years_match = re.search(years_pattern, resume_lower)
        experience_years = int(years_match.group(1)) if years_match else 3
        
        # Calculate job match if job description provided
        job_match_score = 0.0
        missing_keywords = []
        if job_description:
            job_lower = job_description.lower()
            job_skills = [skill for skill in technical_skills if skill in job_lower]
            
            if job_skills:
                matches = len(set(found_technical) & set(job_skills))
                job_match_score = matches / len(job_skills)
                missing_keywords = list(set(job_skills) - set(found_technical))[:5]
        
        # Generate intelligent summary
        role_indicators = ['engineer', 'developer', 'analyst', 'manager', 'lead', 'senior', 'architect']
        detected_role = next((role for role in role_indicators if role in resume_lower), 'professional')
        
        summary_parts = [
            f"Experienced {detected_role}",
            f"with {experience_years}+ years of experience" if experience_years > 1 else "with professional experience",
        ]
        
        if found_technical[:3]:
            summary_parts.append(f"skilled in {', '.join(found_technical[:3])}")
        
        if 'leadership' in found_soft or 'lead' in resume_lower:
            summary_parts.append("with leadership experience")
            
        summary = ' '.join(summary_parts) + '.'
        
        # Calculate ATS score based on content quality
        ats_score = 70  # Base score
        if '@' in resume_content: ats_score += 5  # Has email
        if any(phone in resume_content for phone in ['phone', '(', '-', '+']): ats_score += 5  # Has phone
        if len(found_technical) >= 5: ats_score += 10  # Good technical skills
        if experience_years >= 3: ats_score += 5  # Good experience
        if any(word in resume_lower for word in ['achieved', 'improved', 'increased', 'reduced', 'led']): ats_score += 5
        
        return {
            "resume_summary": summary,
            "identified_skills": found_technical + found_soft[:3],
            "skill_categories": {
                "technical": found_technical[:8],
                "soft": found_soft[:3],
                "industry": []
            },
            "ats_compatibility_score": min(ats_score, 95),
            "strengths": [
                f"Strong technical background in {found_technical[0]}" if found_technical else "Professional experience",
                f"{experience_years}+ years of experience" if experience_years > 1 else "Relevant experience"
            ],
            "improvement_areas": [
                "Add more quantified achievements",
                "Include relevant keywords" if missing_keywords else "Consider adding industry-specific terms"
            ],
            "keyword_suggestions": missing_keywords[:5] if missing_keywords else ["Add metrics and numbers", "Include action verbs"],
            "job_match_score": job_match_score
        }

# Service instance
optimizer_service = OllamaResumeOptimizer() if OLLAMA_AVAILABLE else None

# API Routes
@api_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    resume_text: str = Form(...),
    job_description: Optional[str] = Form(None),
    job_title: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    template_id: Optional[str] = Form(None)
):
    """
    Analyze resume and provide optimization suggestions
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        if not optimizer_service:
            raise HTTPException(status_code=500, detail="AI service not available. Please ensure Ollama is installed and running.")
        
        # Validate input
        if len(resume_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Resume content too short (minimum 50 characters)")
        
        # Combine job info if provided
        job_info = None
        if job_description:
            job_parts = [job_description]
            if job_title:
                job_parts.insert(0, f"Job Title: {job_title}")
            if company:
                job_parts.insert(1, f"Company: {company}")
            job_info = "\n".join(job_parts)
        
        # Perform AI analysis
        analysis_result = await optimizer_service.analyze_resume(resume_text, job_info)
        suggestions = await optimizer_service.generate_optimization_suggestions(resume_text, job_info)
        
        # Process suggestions
        optimization_suggestions = []
        for suggestion in suggestions[:10]:  # Limit to top 10 suggestions
            try:
                optimization_suggestions.append(OptimizationSuggestion(
                    section=suggestion.get('section', 'General'),
                    original_text=suggestion.get('original_text', '')[:200],  # Truncate for response
                    optimized_text=suggestion.get('optimized_text', ''),
                    reason=suggestion.get('reason', ''),
                    priority=suggestion.get('priority', 'medium')
                ))
            except Exception as e:
                print(f"Error processing suggestion: {e}")
                continue
        
        # Build skill analysis
        skill_analysis = SkillAnalysis(
            identified_skills=analysis_result.get('identified_skills', []),
            missing_skills=[],  # Will be calculated if job description provided
            skill_match_score=analysis_result.get('job_match_score', 0.0),
            skill_categories=analysis_result.get('skill_categories', {})
        )
        
        # Calculate missing skills if job provided
        if job_info and analysis_result.get('keyword_suggestions'):
            skill_analysis.missing_skills = analysis_result.get('keyword_suggestions', [])[:10]
        
        processing_time = time.time() - start_time
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow(),
            resume_summary=analysis_result.get('resume_summary', 'Analysis completed'),
            job_match_score=analysis_result.get('job_match_score'),
            skill_analysis=skill_analysis,
            optimization_suggestions=optimization_suggestions,
            processing_time=processing_time,
            ats_score=float(analysis_result.get('ats_compatibility_score', 75)),
            template_preserved=True
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.post("/optimize-section")
async def optimize_section(
    section_text: str = Form(...),
    section_type: str = Form(default="general"),
    job_requirements: Optional[str] = Form(None)
):
    """
    Optimize a specific resume section
    """
    try:
        if not optimizer_service:
            raise HTTPException(status_code=500, detail="AI service not available")
        
        optimization_prompt = f"""
        Optimize this resume section to be more effective and ATS-friendly:

        SECTION TYPE: {section_type}
        CURRENT TEXT: {section_text}
        {f"JOB REQUIREMENTS: {job_requirements}" if job_requirements else ""}

        Provide an improved version that:
        1. Uses stronger action verbs
        2. Includes relevant keywords
        3. Quantifies achievements where possible
        4. Is ATS-optimized
        5. Maintains truthfulness

        Respond with just the optimized text, no explanations.
        """
        
        response = ollama.generate(
            model=optimizer_service.model_name,
            prompt=optimization_prompt,
            options={
                'temperature': 0.4,
                'num_predict': 300
            }
        )
        
        optimized_text = response.get('response', section_text).strip()
        
        return {
            "original": section_text,
            "optimized": optimized_text,
            "section_type": section_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@api_router.get("/health")
async def health_check():
    """Check service health"""
    ollama_status = False
    
    if OLLAMA_AVAILABLE:
        try:
            ollama.list()
            ollama_status = True
        except:
            pass
    
    return {
        "status": "healthy" if ollama_status else "degraded",
        "timestamp": datetime.utcnow(),
        "services": {
            "ollama": ollama_status,
            "api": True
        },
        "version": "1.0.0"
    }

@api_router.post("/template")
async def save_template(
    template_data: str = Form(...),
    template_name: str = Form(default="My Template")
):
    """
    Save resume template (for browser storage)
    """
    template_id = str(uuid.uuid4())
    
    # Parse template structure
    structure = {
        "sections": [],
        "formatting": {
            "detected_sections": len(template_data.split('\n\n')),
            "word_count": len(template_data.split()),
            "has_contact_info": any(indicator in template_data.lower() for indicator in ['email', 'phone', '@'])
        }
    }
    
    template_info = TemplateInfo(
        template_id=template_id,
        original_content=template_data[:1000],  # Store first 1000 chars for reference
        structure=structure,
        created_at=datetime.utcnow()
    )
    
    return {
        "template_id": template_id,
        "template_name": template_name,
        "structure": structure,
        "message": "Template saved successfully"
    }

# Original routes (keeping for compatibility)
@api_router.get("/")
async def root():
    return {"message": "AI Resume Optimizer API", "version": "1.0.0"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
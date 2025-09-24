#!/usr/bin/env python3
"""
AI Resume Optimizer Backend API Testing
Tests all endpoints including Ollama integration
"""

import requests
import sys
import json
import time
from datetime import datetime

class ResumeOptimizerTester:
    def __init__(self, base_url="https://template-ats-boost.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details="", response_data=None):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test_name": name,
            "success": success,
            "details": details,
            "response_data": response_data
        })

    def test_health_check(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                ollama_status = data.get('services', {}).get('ollama', False)
                
                if ollama_status:
                    self.log_test("Health Check", True, "API and Ollama both healthy", data)
                    return True
                else:
                    self.log_test("Health Check", False, "Ollama service not available", data)
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, f"Request failed: {str(e)}")
            return False

    def test_root_endpoint(self):
        """Test root API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "AI Resume Optimizer API" in data.get('message', ''):
                    self.log_test("Root Endpoint", True, "API responding correctly", data)
                    return True
                else:
                    self.log_test("Root Endpoint", False, "Unexpected response format", data)
                    return False
            else:
                self.log_test("Root Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Root Endpoint", False, f"Request failed: {str(e)}")
            return False

    def test_template_save(self):
        """Test template saving endpoint"""
        try:
            template_data = """
John Doe
Software Engineer
Email: john.doe@email.com
Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development.
Proficient in Python, JavaScript, React, and cloud technologies.

EXPERIENCE
Senior Developer at Tech Corp (2020-2024)
- Developed scalable web applications using React and Node.js
- Implemented CI/CD pipelines reducing deployment time by 50%
- Led team of 4 developers on critical projects

SKILLS
- Programming: Python, JavaScript, TypeScript, Java
- Frameworks: React, Node.js, Django, Flask
- Cloud: AWS, Docker, Kubernetes
- Databases: PostgreSQL, MongoDB, Redis
            """.strip()

            form_data = {
                'template_data': template_data,
                'template_name': 'Test Resume Template'
            }

            response = requests.post(f"{self.api_url}/template", data=form_data, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'template_id' in data and 'structure' in data:
                    self.log_test("Template Save", True, "Template saved successfully", data)
                    return data.get('template_id')
                else:
                    self.log_test("Template Save", False, "Missing required fields in response", data)
                    return None
            else:
                self.log_test("Template Save", False, f"HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.log_test("Template Save", False, f"Request failed: {str(e)}")
            return None

    def test_resume_analysis_basic(self):
        """Test basic resume analysis without job description"""
        try:
            resume_text = """
John Doe
Software Engineer
Email: john.doe@email.com

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development.
Proficient in Python, JavaScript, React, and cloud technologies.

EXPERIENCE
Senior Developer at Tech Corp (2020-2024)
- Developed scalable web applications using React and Node.js
- Implemented CI/CD pipelines reducing deployment time by 50%
- Led team of 4 developers on critical projects

SKILLS
Programming: Python, JavaScript, TypeScript, Java
Frameworks: React, Node.js, Django, Flask
Cloud: AWS, Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis
            """.strip()

            form_data = {
                'resume_text': resume_text,
                'template_id': 'test_template_123'
            }

            print("🔄 Starting basic resume analysis (may take 30-60 seconds)...")
            response = requests.post(f"{self.api_url}/analyze", data=form_data, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['analysis_id', 'resume_summary', 'skill_analysis', 'ats_score', 'processing_time']
                
                missing_fields = [field for field in required_fields if field not in data]
                if not missing_fields:
                    ats_score = data.get('ats_score', 0)
                    skills_found = len(data.get('skill_analysis', {}).get('identified_skills', []))
                    
                    self.log_test("Basic Resume Analysis", True, 
                                f"ATS Score: {ats_score}, Skills Found: {skills_found}", 
                                {k: v for k, v in data.items() if k != 'optimization_suggestions'})
                    return data
                else:
                    self.log_test("Basic Resume Analysis", False, f"Missing fields: {missing_fields}", data)
                    return None
            else:
                self.log_test("Basic Resume Analysis", False, f"HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.log_test("Basic Resume Analysis", False, f"Request failed: {str(e)}")
            return None

    def test_resume_analysis_with_job(self):
        """Test resume analysis with job description"""
        try:
            resume_text = """
John Doe
Software Engineer
Email: john.doe@email.com

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development.
Proficient in Python, JavaScript, React, and cloud technologies.

EXPERIENCE
Senior Developer at Tech Corp (2020-2024)
- Developed scalable web applications using React and Node.js
- Implemented CI/CD pipelines reducing deployment time by 50%
- Led team of 4 developers on critical projects

SKILLS
Programming: Python, JavaScript, TypeScript, Java
Frameworks: React, Node.js, Django, Flask
Cloud: AWS, Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis
            """.strip()

            job_description = """
We are seeking a Senior Full Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and JavaScript
- Experience with React, Node.js, and modern web frameworks
- Knowledge of cloud platforms (AWS preferred)
- Experience with containerization (Docker, Kubernetes)
- Database experience with PostgreSQL and NoSQL databases
- Experience with CI/CD pipelines and DevOps practices
- Strong problem-solving skills and team collaboration

NICE TO HAVE:
- Experience with microservices architecture
- Knowledge of machine learning frameworks
- Experience with GraphQL
- Agile/Scrum methodology experience
            """.strip()

            form_data = {
                'resume_text': resume_text,
                'job_description': job_description,
                'job_title': 'Senior Full Stack Developer',
                'company': 'Tech Innovations Inc',
                'template_id': 'test_template_456'
            }

            print("🔄 Starting targeted resume analysis with job description (may take 60-90 seconds)...")
            response = requests.post(f"{self.api_url}/analyze", data=form_data, timeout=150)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for job match score
                job_match_score = data.get('job_match_score')
                optimization_suggestions = data.get('optimization_suggestions', [])
                
                if job_match_score is not None and len(optimization_suggestions) > 0:
                    self.log_test("Targeted Resume Analysis", True, 
                                f"Job Match: {job_match_score:.1%}, Suggestions: {len(optimization_suggestions)}", 
                                {k: v for k, v in data.items() if k not in ['optimization_suggestions']})
                    return data
                else:
                    self.log_test("Targeted Resume Analysis", False, 
                                f"Missing job match score or suggestions. Match: {job_match_score}, Suggestions: {len(optimization_suggestions)}")
                    return None
            else:
                self.log_test("Targeted Resume Analysis", False, f"HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.log_test("Targeted Resume Analysis", False, f"Request failed: {str(e)}")
            return None

    def test_section_optimization(self):
        """Test section-specific optimization"""
        try:
            section_text = "Experienced software engineer with knowledge of various programming languages."
            job_requirements = "Looking for senior developer with 5+ years Python and React experience"

            form_data = {
                'section_text': section_text,
                'section_type': 'professional_summary',
                'job_requirements': job_requirements
            }

            print("🔄 Testing section optimization...")
            response = requests.post(f"{self.api_url}/optimize-section", data=form_data, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if 'original' in data and 'optimized' in data:
                    optimized_length = len(data.get('optimized', ''))
                    if optimized_length > len(section_text):
                        self.log_test("Section Optimization", True, 
                                    f"Section optimized successfully. Length: {optimized_length} chars", data)
                        return True
                    else:
                        self.log_test("Section Optimization", False, "Optimized text not significantly improved")
                        return False
                else:
                    self.log_test("Section Optimization", False, "Missing required response fields", data)
                    return False
            else:
                self.log_test("Section Optimization", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Section Optimization", False, f"Request failed: {str(e)}")
            return False

    def test_error_handling(self):
        """Test API error handling with invalid inputs"""
        try:
            # Test with too short resume
            form_data = {'resume_text': 'Too short'}
            response = requests.post(f"{self.api_url}/analyze", data=form_data, timeout=30)
            
            if response.status_code == 400:
                self.log_test("Error Handling - Short Resume", True, "Correctly rejected short resume")
            else:
                self.log_test("Error Handling - Short Resume", False, f"Expected 400, got {response.status_code}")

            # Test with empty section optimization
            form_data = {'section_text': ''}
            response = requests.post(f"{self.api_url}/optimize-section", data=form_data, timeout=30)
            
            if response.status_code in [400, 422]:
                self.log_test("Error Handling - Empty Section", True, "Correctly rejected empty section")
                return True
            else:
                self.log_test("Error Handling - Empty Section", False, f"Expected 400/422, got {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Error Handling", False, f"Request failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting AI Resume Optimizer Backend Tests")
        print(f"🌐 Testing API at: {self.api_url}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Basic connectivity tests
        health_ok = self.test_health_check()
        root_ok = self.test_root_endpoint()
        
        if not health_ok:
            print("\n⚠️  WARNING: Ollama service not available - AI features may not work")
            print("   This is expected if Ollama is not installed or running")
        
        # Template management
        template_id = self.test_template_save()
        
        # Core analysis features
        basic_analysis = self.test_resume_analysis_basic()
        targeted_analysis = self.test_resume_analysis_with_job()
        
        # Additional features
        section_opt = self.test_section_optimization()
        error_handling = self.test_error_handling()
        
        # Summary
        total_time = time.time() - start_time
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        
        print("\n" + "=" * 60)
        print(f"📊 TEST SUMMARY")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Time: {total_time:.1f}s")
        
        # Detailed analysis of critical features
        critical_tests = ['Health Check', 'Basic Resume Analysis', 'Targeted Resume Analysis']
        critical_passed = sum(1 for result in self.test_results 
                            if result['test_name'] in critical_tests and result['success'])
        
        print(f"\n🎯 CRITICAL FEATURES: {critical_passed}/{len(critical_tests)} working")
        
        if success_rate >= 80:
            print("✅ Backend API is functioning well!")
            return 0
        elif success_rate >= 60:
            print("⚠️  Backend API has some issues but core features work")
            return 1
        else:
            print("❌ Backend API has significant issues")
            return 2

def main():
    tester = ResumeOptimizerTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())
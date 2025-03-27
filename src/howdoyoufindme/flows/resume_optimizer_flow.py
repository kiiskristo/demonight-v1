# src/howdoyoufindme/flows/resume_optimizer_flow.py

from crewai.flow.flow import Flow, listen, start, FlowState
from crewai import Agent, Crew, Process
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncGenerator
import json
import asyncio
import logging
import re
from ..clean_json import clean_and_parse_json
from ..crew import HowDoYouFindMeCrew
from ..tools.resume_tools import CacheStorageTool

class ResumeOptimizerState(FlowState):
    resume_path: str
    job_description: Optional[str] = None
    additional_info: Optional[str] = None
    resume_analysis: Optional[Dict[str, Any]] = None
    user_profile: Optional[Dict[str, Any]] = None
    job_profile: Optional[Dict[str, Any]] = None
    optimized_resume: Optional[Dict[str, Any]] = None

class ResumeOptimizerFlow(Flow[ResumeOptimizerState]):
    def __init__(self, resume_path: str, job_description: Optional[str] = None, additional_info: Optional[str] = None):
        self.initial_state = ResumeOptimizerState(
            resume_path=resume_path,
            job_description=job_description,
            additional_info=additional_info
        )
        super().__init__()
        self._initialize_crew()

    def _initialize_crew(self):
        """Initialize crew instance with separate crews for each task"""
        self.crew_instance = HowDoYouFindMeCrew()
        
        self.resume_analyzer_crew = Crew(
            agents=[self.crew_instance.resume_analyzer_agent()],
            tasks=[self.crew_instance.analyze_resume_task()],
            process=Process.sequential,
            verbose=True
        )
        
        self.profile_builder_crew = Crew(
            agents=[self.crew_instance.profile_builder_agent()],
            tasks=[self.crew_instance.build_profile_task()],
            process=Process.sequential,
            verbose=True
        )
        
        self.job_researcher_crew = Crew(
            agents=[self.crew_instance.job_researcher_agent()],
            tasks=[self.crew_instance.research_job_task()],
            process=Process.sequential,
            verbose=True
        )
        
        self.resume_optimizer_crew = Crew(
            agents=[self.crew_instance.resume_optimizer_agent()],
            tasks=[self.crew_instance.optimize_resume_task()],
            process=Process.sequential,
            verbose=True
        )

    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract and clean JSON from agent response"""
        try:
            # First try direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Find content between first { and last }
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx + 1]
                    # Remove escaped quotes that might be causing issues
                    json_str = re.sub(r'\\+"', '"', json_str)
                    return json.loads(json_str)
            except Exception:
                try:
                    # Last resort: try clean_and_parse_json
                    return clean_and_parse_json(text)
                except Exception as e:
                    logging.error(f"Failed to parse JSON: {str(e)}\nRaw text: {text[:200]}...")
                    return None

    @start()
    async def analyze_resume(self):
        """Start by analyzing the resume"""
        try:
            result = self.resume_analyzer_crew.kickoff(inputs={"file_path": self.state.resume_path})
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.resume_analysis = data
                    return data
        except Exception as e:
            logging.error(f"Error in analyze_resume: {str(e)}")
        return None

    @listen(analyze_resume)
    async def build_user_profile(self, resume_analysis):
        """Build user profile from resume analysis and additional info"""
        try:
            # Convert resume_analysis to string for JSON serialization if needed
            inputs = {
                "additional_info": self.state.additional_info or ""
            }
            
            # For debugging
            logging.info(f"Building user profile...")
            
            # Store resume analysis in cache first so the agent can retrieve it
            cache_key = "resume_analysis_" + str(hash(str(resume_analysis)))
            cache_tool = CacheStorageTool()
            cache_tool.run("store", cache_key, resume_analysis)
            logging.info(f"Stored resume analysis in cache with key: {cache_key}")
            
            # Add the cache key to the inputs
            inputs["resume_analysis_key"] = cache_key
            
            result = self.profile_builder_crew.kickoff(inputs=inputs)
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.user_profile = data
                    return data
        except Exception as e:
            logging.error(f"Error in build_user_profile: {str(e)}")
            # Log the stack trace for debugging
            import traceback
            logging.error(traceback.format_exc())
        return None

    @listen(build_user_profile)
    async def research_job(self, user_profile):
        """Research job details if job description is provided"""
        if not self.state.job_description:
            return None
            
        try:
            # Store user profile in cache
            cache_key = "user_profile_" + str(hash(str(user_profile)))
            cache_tool = CacheStorageTool()
            cache_tool.run("store", cache_key, user_profile)
            logging.info(f"Stored user profile in cache with key: {cache_key}")
            
            inputs = {
                "job_description": self.state.job_description,
                "user_profile_key": cache_key
            }
            logging.info(f"Researching job with description length: {len(self.state.job_description)}")
            
            result = self.job_researcher_crew.kickoff(inputs=inputs)
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.job_profile = data
                    return data
        except Exception as e:
            logging.error(f"Error in research_job: {str(e)}")
        return None

    @listen(research_job)
    async def optimize_resume(self, job_profile):
        """Create optimized resume based on user profile and job details"""
        try:
            # Store job profile in cache
            job_cache_key = "job_profile_" + str(hash(str(job_profile)))
            user_cache_key = "user_profile_" + str(hash(str(self.state.user_profile)))
            
            cache_tool = CacheStorageTool()
            cache_tool.run("store", job_cache_key, job_profile)
            cache_tool.run("store", user_cache_key, self.state.user_profile)
            
            logging.info(f"Stored profiles in cache with keys: {job_cache_key}, {user_cache_key}")
            
            # Prepare inputs for the task
            inputs = {
                "user_profile_key": user_cache_key,
                "job_profile_key": job_cache_key
            }
            
            logging.info("Optimizing resume with profiles")
            
            result = self.resume_optimizer_crew.kickoff(inputs=inputs)
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.optimized_resume = data
                    return data
        except Exception as e:
            logging.error(f"Error in optimize_resume: {str(e)}")
        return None

    async def stream_optimization(self) -> AsyncGenerator[str, None]:
        """Stream the resume optimization process"""
        try:
            yield self._format_event("status", "Starting resume analysis...")
            await asyncio.sleep(0.1)

            resume_analysis = await self.analyze_resume()
            if resume_analysis:
                yield self._format_event("task_complete", task="resume_analysis", data=resume_analysis)
                yield self._format_event("status", "Building user profile...")
                
                user_profile = await self.build_user_profile(resume_analysis)
                if user_profile:
                    yield self._format_event("task_complete", task="user_profile", data=user_profile)
                    
                    if self.state.job_description:
                        yield self._format_event("status", "Researching job requirements...")
                        job_profile = await self.research_job(user_profile)
                        
                        if job_profile:
                            yield self._format_event("task_complete", task="job_profile", data=job_profile)
                            yield self._format_event("status", "Optimizing resume...")
                            
                            optimized_resume = await self.optimize_resume(job_profile)
                            if optimized_resume:
                                yield self._format_event("task_complete", task="optimized_resume", data=optimized_resume)
                                yield self._format_event("complete", "Resume optimization complete")
                            else:
                                yield self._format_event("error", "Failed to optimize resume")
                        else:
                            yield self._format_event("error", "Failed to research job")
                    else:
                        # If no job description, we still provide the user profile
                        yield self._format_event("complete", "User profile creation complete")
                else:
                    yield self._format_event("error", "Failed to build user profile")
            else:
                yield self._format_event("error", "Failed to analyze resume")
                    
        except Exception as e:
            logging.error(f"Error in stream_optimization: {str(e)}")
            yield self._format_event("error", f"Error during optimization: {str(e)}")

    def _format_event(self, event_type: str, message: str = None, task: str = None, data: Dict = None) -> str:
        """Format an event for SSE streaming"""
        event = {"type": event_type}
        if message:
            event["message"] = message
        if task:
            event["task"] = task
        if data:
            event["data"] = data
        return f"data: {json.dumps(event)}\n\n" 
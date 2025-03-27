# src/howdoyoufindme/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .flows.search_rank_flow import SearchRankFlow
from .flows.resume_optimizer_flow import ResumeOptimizerFlow
from typing import AsyncGenerator, Optional
import asyncio
import requests
import os
import tempfile
import shutil
import uvicorn
import logging
from .crew import HowDoYouFindMeCrew
from .tools.resume_tools import CacheStorageTool, resume_tool
from .clean_json import clean_and_parse_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get reCAPTCHA secret key from environment variable
RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "text/event-stream"]
)

async def search_rank_event_generator(query: str) -> AsyncGenerator[str, None]:
    """Generate SSE events from flow"""
    flow = SearchRankFlow(query)
    async for event in flow.stream_analysis():
        yield event
        await asyncio.sleep(0)

async def resume_optimizer_event_generator(
    resume_path: str, 
    job_description_path: Optional[str] = None, 
    additional_info: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Generate SSE events from resume optimizer flow"""
    # If job description file is provided, read its contents
    job_description = None
    if job_description_path and os.path.exists(job_description_path):
        try:
            with open(job_description_path, 'r') as f:
                job_description = f.read()
        except Exception as e:
            logger.error(f"Error reading job description file: {str(e)}")
    
    flow = ResumeOptimizerFlow(
        resume_path=resume_path,
        job_description=job_description,
        additional_info=additional_info
    )
    async for event in flow.stream_optimization():
        yield event
        await asyncio.sleep(0)

async def verify_recaptcha(token: str):
    """Verify reCAPTCHA token with Google's API"""
    if not RECAPTCHA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="reCAPTCHA secret key not configured")
        
    verification_response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={
            'secret': RECAPTCHA_SECRET_KEY,
            'response': token
        }
    )
    
    verification_result = verification_response.json()
    
    if not verification_result.get('success', False):
        raise HTTPException(status_code=400, detail="reCAPTCHA verification failed")
    
    return True

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}
    
@app.get("/api/search-rank/flow")
async def search_rank_flow(query: str, recaptcha_token: str = None):
    logger.info(f"Search rank flow called with query: {query}")
    # Skip reCAPTCHA verification in development environment
    if os.environ.get("ENVIRONMENT") != "development" and RECAPTCHA_SECRET_KEY:
        if not recaptcha_token:
            raise HTTPException(status_code=400, detail="reCAPTCHA token required")
        
        # Verify reCAPTCHA token
        await verify_recaptcha(recaptcha_token)
    
    return StreamingResponse(
        search_rank_event_generator(query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )

@app.post("/api/resume/analyze")
async def analyze_resume_endpoint(
    resume: UploadFile = File(...)
):
    logger.info(f"Resume analysis called with file: {resume.filename}")
    temp_dir = tempfile.mkdtemp()
    resume_path = os.path.join(temp_dir, resume.filename)

    try:
        # Save the uploaded resume
        with open(resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        logger.info(f"Resume saved to temporary path: {resume_path}")

        # Initialize only the necessary crew components
        crew_instance = HowDoYouFindMeCrew()
        resume_analyzer_crew = Crew(
            agents=[crew_instance.resume_analyzer_agent()],
            tasks=[crew_instance.analyze_resume_task()],
            process=Process.sequential,
            verbose=True
        )

        # Kick off the analysis task
        # Note: Ensure the analyze_resume_task uses 'file_path' as input key
        result = resume_analyzer_crew.kickoff(inputs={"file_path": resume_path})

        analysis_data = None
        if result and result.tasks_output and hasattr(result.tasks_output[0], 'raw'):
            raw_output = result.tasks_output[0].raw
            logger.info(f"Raw analysis output: {raw_output[:500]}...") # Log snippet
            analysis_data = clean_and_parse_json(raw_output) # Use your JSON cleaner

        if not analysis_data:
             logger.error("Failed to parse analysis data from the agent.")
             raise HTTPException(status_code=500, detail="Failed to analyze resume data.")

        # Cache the result
        cache_tool_instance = CacheStorageTool()
        # Use a hash of the data or a UUID for a unique key
        cache_key = f"resume_analysis_{hash(json.dumps(analysis_data, sort_keys=True))}"
        cache_tool_instance.run(action="store", key=cache_key, data=analysis_data)
        logger.info(f"Analysis result cached with key: {cache_key}")

        # Add the cache key to the response
        response_data = {
            "analysis": analysis_data,
            "resume_analysis_key": cache_key
        }

        return response_data

    except Exception as e:
        logger.error(f"Error during resume analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

@app.post("/api/resume/optimize")
async def optimize_resume_flow(
    resume_analysis_key: str = Form(...),
    answers: str = Form(...), # Assuming answers are provided as a single string for now
    job_description: Optional[UploadFile] = File(None),
    additional_info: Optional[str] = Form(None), # Keep this if needed
    recaptcha_token: Optional[str] = Form(None) # Keep recaptcha if needed
):
    logger.info(f"Resume optimization flow called with key: {resume_analysis_key}")
    # ... (Optional: reCAPTCHA verification) ...

    temp_dir = tempfile.mkdtemp()
    job_description_path = None
    try:
        # Save job description file if provided
        if job_description:
            job_description_path = os.path.join(temp_dir, job_description.filename)
            with open(job_description_path, "wb") as buffer:
                shutil.copyfileobj(job_description.file, buffer)
            logger.info(f"Job description saved to: {job_description_path}")

        # Define the generator for the streaming response
        async def event_generator() -> AsyncGenerator[str, None]:
            # Initialize the flow with the key and answers
            flow = ResumeOptimizerFlow(
                resume_analysis_key=resume_analysis_key,
                answers=answers,
                job_description_path=job_description_path, # Pass the path
                additional_info=additional_info
            )
            async for event in flow.stream_optimization():
                yield event
                await asyncio.sleep(0) # Yield control briefly

        # Return the streaming response
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked"
            }
        )
    except Exception as e:
        logger.error(f"Error in optimization flow endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error optimizing resume: {str(e)}")
    finally:
        # Clean up temp directory if job description was saved
        if job_description_path:
             shutil.rmtree(temp_dir, ignore_errors=True)
             logger.info(f"Cleaned up temporary directory: {temp_dir}")

# Run the app with uvicorn when this script is executed directly
if __name__ == "__main__":
    logger.info("Starting the FastAPI server...")
    uvicorn.run("howdoyoufindme.main:app", host="0.0.0.0", port=8000, reload=True)
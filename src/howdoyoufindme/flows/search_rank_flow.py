# src/howdoyoufindme/flows/search_rank_flow.py

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

class SearchRankState(FlowState):
    query: str
    keywords: Optional[Dict[str, Any]] = None
    queries: Optional[Dict[str, Any]] = None
    ranking: Optional[Dict[str, Any]] = None

class SearchRankFlow(Flow[SearchRankState]):
    def __init__(self, query: str):
        self.initial_state = SearchRankState(query=query)
        super().__init__()
        self._initialize_crew()

    def _initialize_crew(self):
        """Initialize crew instance with separate crews for each task"""
        self.crew_instance = HowDoYouFindMeCrew()
        
        self.keyword_crew = Crew(
            agents=[self.crew_instance.keyword_agent()],
            tasks=[self.crew_instance.generate_keywords_task()],
            process=Process.sequential,
            verbose=True
        )
        
        self.query_crew = Crew(
            agents=[self.crew_instance.query_builder_agent()],
            tasks=[self.crew_instance.build_query_task()],
            process=Process.sequential,
            verbose=True
        )
        
        self.ranking_crew = Crew(
            agents=[self.crew_instance.ranking_agent()],
            tasks=[self.crew_instance.ranking_task()],
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
    async def analyze_keywords(self):
        """Start the analysis by processing keywords"""
        try:
            result = self.keyword_crew.kickoff(inputs={"query": self.state.query})
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.keywords = data
                    return data
        except Exception as e:
            logging.error(f"Error in analyze_keywords: {str(e)}")
        return None

    @listen(analyze_keywords)
    async def process_queries(self, keyword_result):
        """Process search queries based on keywords"""
        try:
            result = self.query_crew.kickoff(inputs={"query": self.state.query})
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.queries = data
                    return data
        except Exception as e:
            logging.error(f"Error in process_queries: {str(e)}")
        return None

    @listen(process_queries)
    async def determine_ranking(self, query_result):
        """Determine final rankings"""
        try:
            result = self.ranking_crew.kickoff(inputs={"query": self.state.query})
            if hasattr(result.tasks_output[0], 'raw'):
                data = self._extract_json_from_response(result.tasks_output[0].raw)
                if data:
                    self.state.ranking = data
                    return data
        except Exception as e:
            logging.error(f"Error in determine_ranking: {str(e)}")
        return None

    async def stream_analysis(self) -> AsyncGenerator[str, None]:
        """Stream the analysis process"""
        try:
            yield self._format_event("status", "Starting analysis...")
            await asyncio.sleep(0.1)

            keywords = await self.analyze_keywords()
            if keywords:
                yield self._format_event("task_complete", task="keywords", data=keywords)
                yield self._format_event("status", "Analyzing queries...")
                
                queries = await self.process_queries(keywords)
                if queries:
                    yield self._format_event("task_complete", task="queries", data=queries)
                    yield self._format_event("status", "Determining ranking...")
                    
                    ranking = await self.determine_ranking(queries)
                    if ranking:
                        yield self._format_event("task_complete", task="ranking", data=ranking)
                        yield self._format_event("complete", "Analysis complete")
                    else:
                        yield self._format_event("error", "Failed to process ranking data")
                else:
                    yield self._format_event("error", "Failed to process queries data")
            else:
                yield self._format_event("error", "Failed to process keywords data")
                    
        except Exception as e:
            logging.error(f"Error in stream_analysis: {str(e)}")
            yield self._format_event("error", f"Error during analysis: {str(e)}")

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
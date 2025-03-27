from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from .tools.search_tool import MarketSearchTool
from .tools.resume_tools import resume_tool, job_search_tool, cache_tool
from dotenv import load_dotenv

@CrewBase
class HowDoYouFindMeCrew:
    """HowDoYouFindMe crew for resume optimization and job matching"""

    def __init__(self):
        load_dotenv()
        self.market_search_tool = MarketSearchTool()

    @agent
    def resume_analyzer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_analyzer_agent'],
            tools=[resume_tool, cache_tool],
            llm_config={"temperature": 0.7, "model": "gpt-4o-mini"},
            verbose=True
        )

    @agent
    def profile_builder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['profile_builder_agent'],
            tools=[cache_tool],
            llm_config={"temperature": 0.7, "model": "gpt-4o-mini"},
            verbose=True
        )

    @agent
    def job_researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['job_researcher_agent'],
            tools=[job_search_tool, cache_tool],
            llm_config={"temperature": 0.7, "model": "gpt-4o-mini"},
            verbose=True
        )

    @agent
    def resume_optimizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_optimizer_agent'],
            tools=[cache_tool],
            llm_config={"temperature": 0.0, "model": "gpt-4o-mini"},
            verbose=True
        )

    @task
    def analyze_resume_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_resume_task']
        )

    @task
    def build_profile_task(self) -> Task:
        return Task(
            config=self.tasks_config['build_profile_task']
        )

    @task
    def research_job_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_job_task']
        )

    @task
    def optimize_resume_task(self) -> Task:
        return Task(
            config=self.tasks_config['optimize_resume_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
from langchain.tools import BaseTool
from langchain.utilities import BingSearchAPIWrapper
from pydantic import BaseModel, Field
from typing import Type, ClassVar
import os


class SearchToolInput(BaseModel):
    """Input schema for MarketSearchTool."""
    query: str = Field(
        ...,
        description="Search query to find market information, rankings, or company details."
    )


class MarketSearchTool(BaseTool):
    name: ClassVar[str] = "market_search"
    description: ClassVar[str] = (
        "Use this tool to search for market information, company rankings, "
        "and competitive analysis data. It performs web searches focused on "
        "business and market intelligence."
    )
    args_schema: Type[BaseModel] = SearchToolInput
    bing_search: BingSearchAPIWrapper = None

    def __init__(self):
        super().__init__()
        self.bing_search = BingSearchAPIWrapper(
            bing_subscription_key=os.getenv('BING_SUBSCRIPTION_KEY'),
            bing_search_url="https://api.bing.microsoft.com/v7.0/search"
        )

    def _run(self, query: str) -> str:
        try:
            results = self.bing_search.run(query)
            return results
        except Exception as e:
            return f"Error performing search: {str(e)}"
            
    async def _arun(self, query: str) -> str:
        # Just use the synchronous version for now
        return self._run(query)
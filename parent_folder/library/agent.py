from google.adk.agents import LlmAgent
from .archive.agent import utility_agent
from google.adk.tools.agent_tool import AgentTool

root_agent = LlmAgent(
    name="book_search_agent",
    model="gemini-2.0-flash",
    description="A helpful agent that can search for books and return the top k chunks.",
    instruction="""
    You can use the following tools to search for books:
    - utility_agent: A tool that can search for books and return the top k chunks.
    """,
    tools=[AgentTool(utility_agent)],
)

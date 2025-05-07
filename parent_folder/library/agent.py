from google.adk.agents import Agent
from parent_folder.library.archive.archive import process_pdf_pipeline, interactive_search

# root_agent = Agent(
#     name="weather_time_agent",
#     model="gemini-2.0-flash",
#     description=(
#         "Agent to answer questions about the time and weather in a city."
#     ),
#     instruction=(
#         "You are a helpful agent who can answer user questions about the time and weather in a city."
#     ),
#     tools=[get_weather, get_current_time],
# )

root_agent = Agent(
    name="book_search_agent",
    model="gemini-2.0-flash",
    description=("A helpful agent that can search for books and return the top k chunks."),
    instruction=("You are a helpful agent that can search for books and return the top k chunks."),
    tools=[process_pdf_pipeline, interactive_search],
)

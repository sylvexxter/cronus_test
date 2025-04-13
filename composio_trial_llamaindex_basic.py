import os
import dotenv
#from textwrap import dedent
from composio_llamaindex import Action, App, ComposioToolSet
from composio_llamaindex import App, ComposioToolSet, Action
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from datetime import datetime
from llama_index.core import Settings

GOOGLE_API_KEY = "AIzaSyCdpMuBAsaPWISuYmBQAKOUGplQZ779o-k"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Load environment variables from .env file
dotenv.load_dotenv()
#Settings.llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])
#llm = Groq(model="llama-3.2-3b-preview", api_key=os.environ["GROQ_API_KEY"])
# Settings.llm = OpenAI(model="gpt-4o")
# llm = OpenAI(model="gpt-4o")
Settings.llm = Gemini(model="models/gemini-2.0-flash")
llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)
# Initialize the ComposioToolSet
toolset = ComposioToolSet()

# Get the RAG tool from the Composio ToolSet
tools = toolset.get_tools(apps=[App.GOOGLECALENDAR])

# Retrieve the current date and time
date = datetime.today().strftime("%Y-%m-%d")
timezone = datetime.now().astimezone().tzinfo

# Setup Todo
todo = """
    1PM - 3PM -> Code,
    5PM - 7PM -> Meeting,
    9AM - 12AM -> Learn something,
    8PM - 10PM -> Game
"""

# Define the RAG Agent
prefix_messages = [
    ChatMessage(
        role="system",
        content=(
        """
        You are an AI agent responsible for taking actions on Google Calendar on users' behalf. 
        You need to take action on Calendar using Google Calendar APIs. Use correct tools to run APIs from the given tool-set.
        """
        ),
    )
]
workflow = FunctionAgent(
    name="Agent",
    description="Google calendar agent",
    llm=llm,
    tools=tools,
    prefix_messages=prefix_messages,  # Initial system messages for context
)


async def main():
    response = await workflow.run(
#         user_msg=f"""
# # Book slots according to {todo}. 
# # Properly Label them with the work provided to be done in that time period. 
# # Schedule it for today. Today's date is {date} (it's in YYYY-MM-DD format) 
# # and make the timezone be EST."""
#     )
    user_msg=f"""
    Today's date is {date} (it's in YYYY-MM-DD format). get me today's events and tell me how long i'm busy for.
    """
    )
    print(response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
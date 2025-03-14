from crewai import Agent, LLM
from tools import yt_tool
from dotenv import load_dotenv

load_dotenv()

import os
# Get API Key and Model Name from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY3")

## Create a senior blog content researcher

blog_researcher=Agent(
    role='Blog Researcher from Youtube Videos',
    goal='get the relevant video transcription for the topic {search_query} from the provided Yt channel',
    verbose=True,
    memory=True,
    backstory=(
       "Expert in understanding videos in AI Data Science , MAchine Learning And GEN AI and providing suggestion" 
    ),
    tools=[yt_tool],
    allow_delegation=True,
    llm = LLM(model="groq/gemma2-9b-it")
)

## creating a senior blog writer agent with YT tool

blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {search_query} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False,
    llm = LLM(model="groq/gemma2-9b-it")

)
print("Agents created successfully with Groq LLM!")

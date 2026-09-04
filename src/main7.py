# this program is with Tavily API using Tavily search tool
# along with structured output using Pydantic 
# Pydantic is a library for data validation and setting default values for the fields

from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


#llm = ChatOllama(temperature=0, model="gemma3:270m")
#llm = ChatOllama(temperature=0, model="llama3.2:3b")
# or: model="functiongemma:270m"
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.6-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello, World! this is Tavily API with structured output using Pydantic")
    result = agent.invoke({"messages": HumanMessage(content="Search for top 3 job postings with Amdocs Billing experience for Architect role using langchain and list their details")})
    #print("Result: ", result)
    print("Answer: ",  result.answer)
    print("Sources: ", result.sources)
    
if __name__ == "__main__":
    main()
    print("Done")


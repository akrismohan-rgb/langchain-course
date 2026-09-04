# this program is with Tavily API using Tavily search tool

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


#llm = ChatOllama(temperature=0, model="gemma3:270m")
llm = ChatOllama(temperature=0, model="llama3.2:3b")
# or: model="functiongemma:270m"
#llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.6-flash")
#tools = [TavilySearch()]
tools = [TavilySearch(include_domains=["naukri.com"])]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello, World! this is Tavily API")
    result = agent.invoke({"messages": HumanMessage(content="Identify top 5 job listings along with the links posted in last 1 week with Amdocs Billing experience for Architect role ")})
    print("Result: ", result)
    
if __name__ == "__main__":
    main()
    print("Done")


# this program is with Tavily API using custom search tool

from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
tavily_client = TavilyClient()


@tool
def search(query: str) -> str:
    """Search the web for information
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching the web for {query}")

    return tavily_client.search(query=query)

#llm = ChatOllama(temperature=0, model="gemma3:270m")
llm = ChatOllama(temperature=0, model="llama3.2:3b")
# or: model="functiongemma:270m"
#llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.6-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello, World! this is Tavily API")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Pune?")})
    print(result)
    
if __name__ == "__main__":
    main()
    print("Done")


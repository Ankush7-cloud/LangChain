from typing import List
from pydantic import BaseModel, Field
from traceback import StackSummary
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
#tavily = TavilySearch(api_key="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")

class Source(BaseModel):
    url:str=Field(description="The url of the source")
class AgentResponse(BaseModel):
    answer:str=Field(description="The agent's answer to he query")
    sources:List[Source]=Field(default_factory=list,description="List of sources used to generatethe answer")


#llm = ChatOpenAI()
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:3b")

tools = [TavilySearch(tavily_api_key="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")]
agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":[HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their detail")]})
    print(result)
 

if __name__ == "__main__":
    main()

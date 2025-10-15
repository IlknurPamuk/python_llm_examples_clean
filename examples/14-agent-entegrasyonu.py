# Örnek 21 : agent entegrasyonu

'''
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()
taviy_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o")

def tavily_search(query: str):
    url = "https://api.tavily.com/search"
    headers = {"Authorization": "Bearer tv_XXXXXXX"}
    resp = requests.post(url, headers=headers, json={"q": query})
    return resp.json()

def hesapla(exp):
    return eval(exp)
tools = [
    Tool(name="hesaplama", func=hesapla, description="Matematik işlemleri yapar"),
    Tool(name="Tavily arama", func=tavily_search, description="Tavily ile internetten arama yapar.")
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)
print(agent.run("200 * 10 sonucu hesaplar mısın ?"))
print(agent.run("Bugün Türkiye'nin enflasyon oranı nedir"))

'''


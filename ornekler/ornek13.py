# Örnek 20 : agent
'''
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

def hesapla(expression):
    return eval(expression)

tools = [
    Tool(name="Hesaplama", func=hesapla, description="matematik işlemleri yapar"),
    Tool(name="web arama", func=DuckDuckGoSearchRun().run, description="internetten blgi alır")
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-destription", verbose=True)
print(agent.run("12 * 8 kaç eder?"))
print(agent.run("Türkiye de en çok kullanılan kız isimleri?"))
'''


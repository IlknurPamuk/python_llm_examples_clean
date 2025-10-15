# Örnek 15 : internetten veri çekmek
'''
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="web arama",
        func=search.run,
        description="İnternetten veri aramak için kullanılır"
    )
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)
cevap = agent.run("Türkiye'nin 2023 nüfusu nedir?")
print("cevap:", cevap)
'''


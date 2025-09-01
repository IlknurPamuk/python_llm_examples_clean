# Örnek 2 : llm orkestrasyonu
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

def toplama(x,y):
    return int(x) + int(y)

tools = [
    Tool(
        name="toplama araci",
        func=lambda x: toplama(*[s.strip(" ()") for s in x.split(",")]),
        description="iki sayiyi toplayan araçtır."
    )
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")
print(agent.run("12 ile 8'i topla ve sonucu bana anlat"))
'''

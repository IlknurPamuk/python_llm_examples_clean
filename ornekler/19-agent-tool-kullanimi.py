# Örnek 26 : agent tool kullanımı
'''
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

tools = [PythonREPLTool()]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

print(agent.run("25'in karesini hesapla"))
'''


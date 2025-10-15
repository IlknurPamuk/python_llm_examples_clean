# Örnek 22 : agent memory ####
'''
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

def hesapla(expression):
    return eval(expression)

tools = [
    Tool(name="hesaplama", func=hesapla, description="matematiksel işlemler yapar.")

]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
print(agent.run("Merhaba ben İlknur"))
print(agent.run("12*8 cevabunun söyler misin ?"))
print(agent.run("İsmimi hatırlıyor musun?"))
'''

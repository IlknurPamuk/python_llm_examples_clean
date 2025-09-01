# Örnek 30 : Tavily ile arama
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv 

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)

search = TavilySearch(max_results=2)
tools = [search]

# LangGraph'ta sadece model ve tools gerekiyor
agent = create_react_agent(
    model,
    tools
)

config = {"configurable": {"thread_id": "abc123"}}

if __name__ == "__main__":
    print("🤖 chat başlatıldı. Çıkmak için 'exit' e basın")
    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            print("başarıyla çıkış yapıldı...")
            break

        response = agent.invoke(
            {"input": user_input},
            config=config
        )
        print(response)


'''

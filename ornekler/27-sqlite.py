# Örnek 33 : SQLite
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

 
load_dotenv()

 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


search = TavilySearch(max_results=2)
tools = [search]


agent = create_react_agent(llm, tools)


config = {"configurable": {"thread_id": "test123"}}


if __name__ == "__main__":
    print("🤖 Agent test ediliyor...\n")
    
    
    response1 = agent.invoke(
        {"messages": [("user", "Merhaba! Sen kimsin?")]}, 
        config=config
    )
    print("Cevap 1:", response1["messages"][-1].content)
    print("-" * 50)
    
    
    response2 = agent.invoke(
        {"messages": [("user", "Dün bana ne demiştin?")]}, 
        config=config
    )
    print("Cevap 2:", response2["messages"][-1].content)
    print("-" * 50)
    
    
    response3 = agent.invoke(
        {"messages": [("user", "Türkiye'nin başkenti nedir?")]}, 
        config=config
    )
    print("Cevap 3:", response3["messages"][-1].content)

    '''
'''

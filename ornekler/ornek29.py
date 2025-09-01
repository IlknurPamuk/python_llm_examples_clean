# Örnek 29 :
'''
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
 

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

search = TavilySearchResults(max_results=2)
tools = [search]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True

)
if __name__ == "__main__":
    print("🤖chatbota hosşgeldiniz çıkmak için 'exit' yazın.")
    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print(response)
        '''
'''


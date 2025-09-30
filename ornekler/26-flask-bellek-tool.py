# Örnek 32 : Flask, Bellek, Tool
'''
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def hesapla(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Hata: {str(e)}"

calculator = Tool(
    name="calculator",
    func=hesapla,
    description="Matematiksel ifadeleri hesaplar."
)
search = DuckDuckGoSearchRun()
tools = [calculator, search]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message")
        result = agent.run(user_input)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    '''
'''

# Örnek 31 : Flask API
'''
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI()

 
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

custom_functions = {
    "add": add_numbers,
    "multiply": multiply_numbers
}

@app.route("/start", methods=["GET", "POST"])
def chat():
    
    if request.method == "GET":
        return "Start endpoint GET isteği çalıştı!"

    
    if request.method == "POST":
        try:
            data = request.get_json()
            user_message = data.get("message", "")
            function_call = data.get("function", None)


           
            if function_call and function_call in custom_functions:
                args = data.get("args", {})
                result = custom_functions[function_call](**args)
                return jsonify({"result": result})

             
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Sen bir yardımcı asistansın"},
                    {"role": "user", "content": user_message}
                ]
            )
            return jsonify({"response": response.choices[0].message.content})
            
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
'''

### Örnek 32 : Flask, Bellek, Tool
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

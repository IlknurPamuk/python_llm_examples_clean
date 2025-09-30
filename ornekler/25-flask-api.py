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


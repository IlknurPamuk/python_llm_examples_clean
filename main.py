
 
'''
# Örnek 1: llm nasıl çalışır?

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "sen bir bilgi asistanisin."},
        {"role": "user", "content": "Türkiyenin başkenti neresi, json formatinda yaz."},

    ],
    response_format={"type": "json_object"}
)
data = json.loads(response.choices[0].message.content)
print(data)
print(data.get("cevap"))
print(data.get("kaynak", "belirtilmemiş"))

'''


 
'''
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
# Örnek 3 : langchain
'''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("Soru: {soru}")

sonuc = llm.invoke(prompt.format(soru="Türkiye'nin en büyük yüz ölçümüne sahip şehri hangisidir?"))

print("model cevabı:", sonuc.content)

'''

# Örnek 4 : langchain çeviri
'''
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

translate_prompt = ChatPromptTemplate.from_template("Çevir (Türkçe -> ingilizce): {soru}")
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

answer_prompt = ChatPromptTemplate.from_template("Question: {text}")
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

overall_chain = SimpleSequentialChain(chains=[translate_chain, answer_chain])

soru = "Dünyanın en yüksek dağı hangsidir?"
print("sonuç:", overall_chain.run(soru))
'''

# Örnek 5 : mesajı hafızaya kaydetmek
'''
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm =  ChatOpenAI(model="gpt-4o")

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory)

print(conversation.run("merhaba ben ilknur"))
print(conversation.run("beni hatırlıyor musun?"))
'''

# Örnek 6: json formatında cvb alma
'''
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
schemas = [
    ResponseSchema(name="cevap", description="sorunun cevabı"),
    ResponseSchema(name="kaynak", description="bilgi kaynağı")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instr = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    "Soru: {soru}\nCevabı şu formatta ver: {format_instr}"
)
chain = LLMChain(llm=llm, prompt=prompt)

result= chain.run({"soru": "Türkiye'nin başeknti neredir?", "format_instr": format_instr})
print(parser.parse(result))
'''

# Örnek 7 : prompt
'''

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("kullanıcı: {soru}\nAsistan:")

soru1 = llm.invoke(prompt.format(soru="dünaynın en büyük okyanusu hangisidir"))
soru2 = llm.invoke(prompt.format(soru="pyhton da llm yazma kuralları nedir?"))

print("cevap1:", soru1.content)
print("cevap2:", soru2.content)

'''



# Örnek 8 : langserve
'''
from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Soru: {soru}")
chain = prompt | llm

add_routes(app, chain, path="/chat")
'''

# Örnek 9 : langchain
'''
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["LANGCHAIN_TRACİNG_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lc-123456"

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Soru: {soru}")
chain = prompt | llm

print(chain.invoke("LLM orkestrasyonu nedir?"))
'''

# Örnek 10 : mesajlaşma tarihi
'''
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain 
from datetime import datetime
from dotenv import load_dotenv
 
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template(
    "Tarih: {tarih}\nKullanıcı: {soru}\nAsistan:"
)
chain = prompt | llm

tarih = datetime.now().strftime("%y-%m-%d %H:%M:%S")

cevap = chain.invoke({"tarih": tarih, "soru": "Bugün hava nasıl?"})
print(cevap)

'''

# Örnek 11 : mesaj geçmişi
'''
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(llm=llm, memory=memory)

conversation.run("benim adım ilknur")
conversation.run("doğum yılımı ve nerde doğduğumu hatırlıyormusun?")
print("\n----Konuşma Geçmişi----")

for msg in memory.chat_memory.messages:
    print(f"{msg.type.upper()}: {msg.content}")

    '''


# Örnek 12 : streaming ve memory
'''
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain 
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

llm = ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamHandler()])
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
print("\n----Streaming Cevap----")
conversation.run("yapay zekann geleceğini anlat")
'''

# Örnek 13 : vektörstore
'''
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [
    "Ankara Türkiyenin başkentidir.",
    "İstanbul Türkiye'nin en büyük şeridir.",
    "İzmir ege bölgesinde bir liman kentidir.",    
]
vectorstore = FAISS.from_texts(texts, embeddings)
print("vector sayısı:", vectorstore.index.ntotal)

'''

# Örnek 14 : rag
'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff" #belgeleri topla + cevapla
)

cevap = qa_chain.run("Tüekiyenin en kalabalık şehri hangisidir?")
print("cevap:", cevap)
'''

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
        description="internetten veri aramak için kullanılır"
    )
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)
cevap = agent.run("Türkiyenin 2023 nüfusu nedir?")
print("cevap:", cevap)
'''

# Örnek 16 : vektörize ve text bölme
'''
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
text = """
yapay zeka modelelri, büyük veri setleri üzerinde eğitilir.
transformer mimarisi, dikkat (attention) mekanizması ile çalışır.
LLM ler insan dilini anlamak ve üretmek için  kullanılır.
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
docs = splitter.split_text(text)
print("parçalar:", docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(docs, embeddings)
print("toplam vektör:", vectorstore.index.ntotal)
'''
# Örnek 17 : hub
'''
from langchain import hub
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = hub.pull("rlm/rag-prompt")#özetleme yapar
llm = ChatOpenAI(model="gpt-4o")

chain = prompt | llm
cevap = chain.invoke({ 
    "context": "langchain LLM'lerle çalışma süreci hakkında bilgiler.",
    "question": "Bu sürecci nasıl kolaylaştır?"
})
print("özet:", cevap.content)
'''

# Örnek 18 : streaming
'''
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
load_dotenv()
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

 
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamHandler()],
    temperature=0.7

)
llm.invoke("yapay zekanın gelecekteki en büyük etkilerini açıkla")
print("\n\nTam cevap:", cevap.content)
'''


# Örnek 19 : rag
'''
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "ankara Türkiye'nin başkrntidir.",
    "istanbul türkiyenin en kalabalık şehridir.",
    "everest dünyanın en yüksek dağıdır"
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
llm = ChatOpenAI(model="gpt-4o")
assistant = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

print("Soru:", "Türkiyenin başkenti neresidir")
print("asistan:", assistant.run("türkiyenin başkenti neresi?"))
'''

# Örnek 20 : agent
'''
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

def hesapla(expression):
    return eval(expression)

tools = [
    Tool(name="Hesaplama", func=hesapla, description="matematik işlemleri yapar"),
    Tool(name="web arama", func=DuckDuckGoSearchRun().run, description="internetten blgi alır")
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-destription", verbose=True)
print(agent.run("12 * 8 kaç eder?"))
print(agent.run("Türkiye de en çok kullanılan kız isimleri?"))
'''

# Örnek 21 : agent entegrasyonu

'''
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()
taviy_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o")

def tavily_search(query: str):
    url = "https://api.tavily.com/search"
    headers = {"Authorization": "Bearer tv_XXXXXXX"}
    resp = requests.post(url, headers=headers, json={"q": query})
    return resp.json()

def hesapla(exp):
    return eval(exp)
tools = [
    Tool(name="hesaplama", func=hesapla, description="matematik işlemleri yapar"),
    Tool(name="Tavily arama", func=tavily_search, description="Tavily ile internetten arama yapar.")
]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)
print(agent.run("200 * 10 kaç eder?"))
print(agent.run("Bugün Türkiye'nin enflasyon oranı nedir"))

'''

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
print(agent.run("merhaba ben ilknur"))
print(agent.run("12*8 kaç eder"))
print(agent.run("beni hatırlıyor musun"))
'''
# Örnek 23 : retrieval notlandırıcısı
'''
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

retrieval_eval = load_evaluator("context_qa", llm=llm)
expected="Ankara"
predicted_docs = ["istanbul türkiyenin en büyük şehridirç.", "Ankara Türkiye'nin başkentidir"]
result = retrieval_eval.evaluate_strings(
    prediction=";".join(predicted_docs),
    reference=expected, 
    input="Türkiye'nin başkenti neresidr"

)
print("Sonuç:",result)
'''

# Örnek 24 : halisinasyon notlandırıcısı
'''
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")
custom_criteria = {
    "hallucination": "Cevap, verilen bağlamda yer almayan veya yanlış bilgilerle uyduruyor mu?"
}

hallucination_eval = load_evaluator(
    "criteria",
    criteria=custom_criteria,
    llm=llm
)

context = "ankara türkiyenin başkentidir."
answer = "Türkiye'nin başkenti İstanbul'dur."
result = hallucination_eval.evaluate_strings(
    prediction=answer,
    input="Türkiye'nin başkenti neresidir?",
    reference=context
)

print("sonuç:", result)
'''


# Örnek 25 : cevap notlandırıcısı
'''
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

answer_eval = load_evaluator("qa", llm=llm)

question = "Türkiye'nin en az nüfüsa sahip ili neresidir?"
answer = "Türkiye'nin en az nüfusa sahip ili Bayburt'dur."
result = answer_eval.evaluate_strings(
    prediction=answer,
    input=question,
    reference="Bayburt"

)
print("değerlendirme:", result)
'''
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

# Örnek 27 : retrieval
'''
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
  "LLM diğer görevlerin aynı sıra metin tanıyıp üretbilen bir tür AI programıdır",
  "LLM' ler büyük veri kümleri üzerinde eğitilir.",
  "LLM'ler makina çğrenmesi üzerinde kuruludur."  

]
db = FAISS.from_texts(docs, embeddings)
retriever = db.as_retriever()
llm = ChatOpenAI(model="gpt-4o")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

print(qa.run("LLM'ler de hangi öğrenme yöntemi uygulnır?"))
'''

# Örnek 28 : cevap kontrol
'''
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

criteria = {
    "correctness": "cevap doğru bilgi içeriyormu"
}
evaluator = load_evaluator("criteria", criteria=criteria, llm=llm)

context = "penguenler kutup bölegelerinde yaşarlar"
answer = "penguenler çölde yaşarlar."
result = evaluator.invoke({
    "input": "Penguenler hangi bölgelerde yaşar",
    "output": answer,
    "reference": context
})

print(result)
'''
'''
from cleverbotfree import Cleverbot

@Cleverbot.connect
def chat(bot, user_prompt, bot_prompt):
    while True:
        user_input = input(user_prompt)
        if user_input == "quit":
            break
        reply = bot.single_helloexchange(user_input)#text ister
        print(bot_prompt, reply)
    bot.close()

chat("You:", "Cleverbot:")
'''

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
# Örnek 34 : SQLite
from dotenv import load_dotenv
import sqlite3

load_dotenv()
conn = sqlite3.connect("okul.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS ogrenciler(
    id INTEGER PRIMARY KEY AUTOINCREMENT,  --Her öğrenciye otomatik ID ver
    ad TEXT ,                              -- Öğrencinin adı
    soyad TEXT ,                           -- Öğrencinin soyadı
    notu INTEGER                           -- Öğrencinin notu
)
""")
def ogrenci_ekle(ad, soyad, notu):
    cursor.execute("INSERT INTO ogrenciler (ad, soyad, notu) VALUES (?, ?, ?)", (ad, soyad, notu))
    conn.commit()

def ogrencileri_listele():
    cursor.execute("SELECT * FROM ogrenciler")
    ogrenciler = cursor.fetchall()
    for ogrenci in ogrenciler:
        print(ogrenci)

def notları_guncelle(ogrenci_id, yeni_not):
    cursor.execute("UPDATE ogrenciler SET notu = ? WHERE id = ?", (yeni_not, ogrenci_id))
    conn.commit()

def notları_sil(ogrenci_id):
    cursor.execute("DELETE FROM ogrenciler WHERE id = ?", (ogrenci_id,))
    conn.commit()

def listele():
    return ogrenciler

ogrenci_ekle("Ali", "Yılmaz", 85)
ogrenci_ekle("Ayşe", "Kara", 90)
ogrenci_ekle("Mehmet", "Demir", 78)

print("Öğrenciler:", ogrencileri_listele())

notları_guncelle(1, 95)
print("Güncel Liste:", ogrencileri_listele())

ogrenci_ekle("Fatma", "Çelik", 88)
print("Güncellenmiş Öğrenciler:", ogrencileri_listele())

'''


# Örnek 35 : Flask API
'''
from flask import Flask, request, jsonify
import sqlite3
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return "hoşgeldiniz flask sistemi çalışıyor"


def get_db():
    conn = sqlite3.connect("okul.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/ogrenciler", methods=["GET"])
def ogrencileri_listele_api():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ogrenciler")
    ogrenciler = [dict(row) for row in cursor.fetchall()]
    return jsonify(ogrenciler)

@app.route("/ekle", methods=["POST"])
def ekle():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ogrenciler (ad, soyad, notu) VALUES (?, ?, ?)",
                   (data["ad"], data["soyad"], data["notu"]))
    conn.commit()
    return jsonify({"message": "Öğrenci eklendi."})

@app.route("/guncelle/<int:id>", methods=["PUT"])
def guncelle(id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE ogrenciler SET notu = ? WHERE id = ?", (data["notu"])),
    conn.commit()
    return jsonify({"mesaj": "not guncellendi"})

@app.route("/sil/<int:id>", methods=["DELETE"])
def sil(id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ogrenciler WHERE id = ?", (id,))
    conn.commit()
    return jsonify({"mesaj": "öğrenci silindi"})

if __name__ == "__main__":
    app.run(debug=True)
'''

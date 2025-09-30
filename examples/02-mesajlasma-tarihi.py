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


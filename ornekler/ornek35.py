# Örnek 09 : langchain
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


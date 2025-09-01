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


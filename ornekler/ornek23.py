# Örnek 03 : langchain
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


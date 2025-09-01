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


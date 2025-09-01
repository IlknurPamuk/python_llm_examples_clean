# Örnek 07 : prompt
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




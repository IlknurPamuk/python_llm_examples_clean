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


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


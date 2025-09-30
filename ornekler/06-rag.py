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


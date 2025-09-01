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


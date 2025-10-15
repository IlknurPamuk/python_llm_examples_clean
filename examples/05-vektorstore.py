# Örnek 13 : vektörstore
'''
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [
    "Ankara Türkiyenin başkentidir.",
    "İstanbul Türkiye'nin en büyük şehridir.",
    "İzmir ege bölgesinde bir liman kentidir.",    
]
vectorstore = FAISS.from_texts(texts, embeddings)
print("vector sayısı:", vectorstore.index.ntotal)

'''


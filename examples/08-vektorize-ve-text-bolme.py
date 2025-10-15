# Örnek 16 : vektörize ve text bölme
'''
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
text = """
Yapay zeka modelleri, büyük veri setleri üzerinde eğitilir.
Transformer mimarisi, dikkat (attention) mekanizması ile çalışır.
LLM ler insan dilini anlamak ve üretmek için  kullanılır.
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
docs = splitter.split_text(text)
print("parçalar:", docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(docs, embeddings)
print("toplam vektör:", vectorstore.index.ntotal)
'''

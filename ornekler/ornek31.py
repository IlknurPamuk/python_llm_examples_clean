# Örnek 05 : mesajı hafızaya kaydetmek
'''
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm =  ChatOpenAI(model="gpt-4o")

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory)

print(conversation.run("merhaba ben ilknur"))
print(conversation.run("beni hatırlıyor musun?"))
'''


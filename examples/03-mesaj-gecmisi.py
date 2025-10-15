# Örnek 11 : mesaj geçmişi
'''
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(llm=llm, memory=memory)

conversation.run("Benim adım ilknur")
conversation.run("Doğum yılımı ve nerde doğduğumu hatırlıyormusun?")
print("\n----Konuşma Geçmişi----")

for msg in memory.chat_memory.messages:
    print(f"{msg.type.upper()}: {msg.content}")

    '''



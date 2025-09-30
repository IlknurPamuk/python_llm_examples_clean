# Örnek 12 : streaming ve memory
'''
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain 
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

llm = ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamHandler()])
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
print("\n----Streaming Cevap----")
conversation.run("yapay zekann geleceğini anlat")
'''


# Örnek 18 : streaming
'''
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
load_dotenv()
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

 
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamHandler()],
    temperature=0.7

)
llm.invoke("yapay zekanın gelecekteki en büyük etkilerini açıkla")
print("\n\nTam cevap:", cevap.content)
'''



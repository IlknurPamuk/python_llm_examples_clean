# Örnek 28 : cevap kontrol
'''
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

criteria = {
    "correctness": "cevap doğru bilgi içeriyormu"
}
evaluator = load_evaluator("criteria", criteria=criteria, llm=llm)

context = "penguenler kutup bölegelerinde yaşarlar"
answer = "penguenler çölde yaşarlar."
result = evaluator.invoke({
    "input": "Penguenler hangi bölgelerde yaşar",
    "output": answer,
    "reference": context
})

print(result)
'''
'''
from cleverbotfree import Cleverbot

@Cleverbot.connect
def chat(bot, user_prompt, bot_prompt):
    while True:
        user_input = input(user_prompt)
        if user_input == "quit":
            break
        reply = bot.single_helloexchange(user_input)#text ister
        print(bot_prompt, reply)
    bot.close()

chat("You:", "Cleverbot:")
'''


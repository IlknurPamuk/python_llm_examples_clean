# Örnek 25 : cevap notlandırıcısı
'''
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

answer_eval = load_evaluator("qa", llm=llm)

question = "Türkiye'nin en az nüfüsa sahip ili neresidir?"
answer = "Türkiye'nin en az nüfusa sahip ili Bayburt'dur."
result = answer_eval.evaluate_strings(
    prediction=answer,
    input=question,
    reference="Bayburt"

)
print("değerlendirme:", result)
'''

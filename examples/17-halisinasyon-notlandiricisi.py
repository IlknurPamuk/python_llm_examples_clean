# Örnek 24 : halisinasyon notlandırıcısı
'''
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")
custom_criteria = {
    "hallucination": "Cevap, verilen bağlamda yer almayan veya yanlış bilgilerle uyduruyor mu?"
}

hallucination_eval = load_evaluator(
    "criteria",
    criteria=custom_criteria,
    llm=llm
)

context = "Ankara Türkiye'nin başkentidir."
answer = "Türkiye'nin başkenti İstanbul'dur."
result = hallucination_eval.evaluate_strings(
    prediction=answer,
    input="Türkiye'nin başkenti neresidir?",
    reference=context
)

print("sonuç:", result)
'''



# Örnek 23 : retrieval notlandırıcısı
'''
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

retrieval_eval = load_evaluator("context_qa", llm=llm)
expected="Ankara"
predicted_docs = ["istanbul türkiyenin en büyük şehridirç.", "Ankara Türkiye'nin başkentidir"]
result = retrieval_eval.evaluate_strings(
    prediction=";".join(predicted_docs),
    reference=expected, 
    input="Türkiye'nin başkenti neresidr"

)
print("Sonuç:",result)
'''


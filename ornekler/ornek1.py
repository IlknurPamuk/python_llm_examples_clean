# Örnek 1: llm nasıl çalışır?

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "sen bir bilgi asistanisin."},
        {"role": "user", "content": "Türkiyenin başkenti neresi, json formatinda yaz."},

    ],
    response_format={"type": "json_object"}
)
data = json.loads(response.choices[0].message.content)
print(data)
print(data.get("cevap"))
print(data.get("kaynak", "belirtilmemiş"))

'''


 
'''

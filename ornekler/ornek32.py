# Örnek 06: json formatında cvb alma
'''
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
schemas = [
    ResponseSchema(name="cevap", description="sorunun cevabı"),
    ResponseSchema(name="kaynak", description="bilgi kaynağı")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instr = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    "Soru: {soru}\nCevabı şu formatta ver: {format_instr}"
)
chain = LLMChain(llm=llm, prompt=prompt)

result= chain.run({"soru": "Türkiye'nin başeknti neredir?", "format_instr": format_instr})
print(parser.parse(result))
'''


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, TypedDict
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens muito engraçadinho e praieiro."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens sério e montanhista."),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um roteador de consultas que decide se a pergunta do usuário é sobre viagens para praias ou montanhas. Se for sobre praias, encaminhe para o consultor de praias; se for sobre montanhas, encaminhe para o consultor de montanhas."),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def response(pergunta: str):
    rota = roteador.invoke({"query": pergunta})["destino"]
    if rota == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    elif rota == "montanha":
        return cadeia_montanha.invoke({"query": pergunta})
    else:
        return "Desculpe, não consegui identificar o tipo de viagem."
    
print(response("Quais são as melhores praias para surfar no Brasil?"))
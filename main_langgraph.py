from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
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

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query":estado["query"]}, config=config)}

async def na_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query":estado["query"]}, config=config)}

async def na_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query":estado["query"]}, config=config)}

def escolher_no(estado: Estado) -> Literal["na_praia", "na_montanha"]:
    if estado["destino"]["destino"] == "praia":
        return "na_praia"
    else:
        return "na_montanha"
    
grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("na_praia", na_praia)
grafo.add_node("na_montanha", na_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("na_praia", END)
grafo.add_edge("na_montanha", END) 

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {
            "query": "Quais são os melhores destinos para férias?"
        }
    )
    print(f"IA: {resposta['resposta']}\n")

asyncio.run(main())
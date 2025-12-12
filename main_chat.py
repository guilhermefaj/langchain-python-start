import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente especializado em sugerir cidades para turistas com base em seus interesses."),
        ("placeholder", "{history}"),
        ("user", "{query}")
    ]
)

chain = prompt_sugestao | modelo | StrOutputParser()

memory = {}
session = "aula_langchain_alura"

def get_history(session: str):
    if session not in memory:
        memory[session] = InMemoryChatMessageHistory()
    return memory[session]

perguntas = [
    "Me sugira as melhoras cidades do mundo para fãs de videogames e tecnologia.",
    "E no Brasil?"
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_history,
    input_messages_key="query",
    history_messages_key="history"
)

for uma_pergunta in perguntas:
    response = cadeia_com_memoria.invoke(
        {
            "query": uma_pergunta
        },
        config={"session_id": session}
    )
    print(f"Usuário: {uma_pergunta}")
    print(f"IA: {response}\n")


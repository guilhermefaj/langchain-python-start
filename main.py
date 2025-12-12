from langchain_openai import ChatOpenAI          # Importa a classe ChatOpenAI da biblioteca langchain_openai
from dotenv import load_dotenv                   # Permite carregar variáveis de ambiente a partir de um arquivo .env
import os                                        # Puxar as variáveis de ambiente do sistema operacional

load_dotenv()                                    # Carrega as variáveis de ambiente do arquivo .env
api_key = os.getenv("OPENAI_API_KEY")            # Obtém a chave da API da OpenAI das variáveis de ambiente

numero_dias = 7
numero_criancas = 2
tipo_atividade = "aventura e natureza"

prompt = f"crie um roteiro de viagem de {numero_dias} dias para {numero_criancas} crianças com atividades de {tipo_atividade}."

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

response = modelo.invoke(prompt)          # Gera a resposta do modelo com base no prompt fornecido

print(response.content)                           # Imprime a resposta gerada pelo modelo
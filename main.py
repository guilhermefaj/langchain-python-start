from langchain_openai import ChatOpenAI          # Importa a classe ChatOpenAI da biblioteca langchain_openai
from langchain.prompts import PromptTemplate     # Importa a classe PromptTemplate da biblioteca langchain.prompts
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel 
from langchain.globals import set_debug
from dotenv import load_dotenv                   # Permite carregar variáveis de ambiente a partir de um arquivo .env
import os                                        # Puxar as variáveis de ambiente do sistema operacional

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar"),
    motivo: str = Field("O motivo pelo qual a cidade é recomendada")

class Restaurantes(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar"),
    Restaurantes: str = Field("Os melhores restaurantes na cidade recomendada")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira os melhores restaurantes na cidade {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="""
    Sugira atividades culturais na cidade {cidade}.
    """
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

cadeia1 = prompt_cidade | modelo | parseador_destino
cadeia2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia1 | cadeia2 | cadeia3)

response = cadeia.invoke(
    {
        "interesse" : "videogames e tecnologia"
    }
)

print(response)
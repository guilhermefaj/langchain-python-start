# Langchain e Python: criando ferramentas com a LLM OpenAI

## Guias de configuração

siga os passos abaixo para configurar seu ambiente e utilizar os scripts do projeto.

### 1. Criar e ativar ambiente virtual

**Windows**
```bash
python -m venv langchain
langchain\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv langchain
source langchain\bin\activate
```

### 2. Instalar Dependências

Utilize o comando abaixo para instalar as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 3. Configurar chave de API da OpenAI

Crie ou edite o arquivo `.env` adicionando sua chave de API da OpenAI:

```bash
OPENAI_API_KEY="sk-gnsj-T3a7tcKa..."
```
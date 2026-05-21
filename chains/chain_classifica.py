from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import  ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

#  Instanciar o modelo
model_classificador = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


class ClassificaEntrada(BaseModel):
    opcao: int = Field(description="Defina 1 se a pergunta do usuário solicitar informações ou orientações sobre a Faculdade de Direito da Universidade kimpa vita. \
Defina 2  se for saudações ou temas que não são referentes a Faculddade de Direito da universidade kimpa vita.\
")

#  Criar o parser e preparar as instruções
parser_classifica = PydanticOutputParser(pydantic_object=ClassificaEntrada)


instrucoes_limpas = parser_classifica.get_format_instructions().replace("{", "{{").replace("}", "}}")

#  Prompt com as instruções já "escapadas"
sys_prompt_rota = f"""Você é um especialista em classificação. Você receberá perguntas do usuário e precisará classificá-las da melhor forma entre as opções estabelecidas.

REGRAS:
1. Retorne APENAS um objeto JSON válido, sem texto adicional.
2. Se a pergunta for sobre a Faculdade de Direito da Universidade Kimpa Vita, defina "opcao": 1.
3. Se for uma saudação, tema não relacionado, pergunta inapropriada ou qualquer coisa que NÃO seja sobre a Faculdade de Direito, defina "opcao": 2.
4. NUNCA recuse responder. A sua única tarefa é retornar o JSON com a classificação.

{instrucoes_limpas}
"""

#  Criar o Template 
rota_prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt_rota),
    ("human", "{input}")
])


chain_de_roteamento = rota_prompt_template | model_classificador | parser_classifica



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnableParallel
from langchain_groq import  ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()



model_fora_do_tema = ChatGroq( model="llama-3.1-8b-instant",
                               temperature=0,
                               groq_api_key=os.getenv("GROQ_API_KEY")
                               )


# Criando o ChatPromptTemplate que vai agir como um personal para tirar dúvidas do usuário
sys_prompt_fora_do_tema = """\
Você é um assistente virtual oficial da Faculdade de Direito da Universidade Kimpa Vita.

REGRAS PARA PERGUNTAS FORA DO CONTEXTO DA FACULDADE:
- Se o usuário fez uma saudação (ex.: "olá", "bom dia"), responda de forma calorosa e educada, e em seguida explique que você está aqui para ajudar com informações sobre a Faculdade de Direito.
- Se o usuário perguntar algo que NÃO seja relacionado à Faculdade de Direito (culinária, engenharia, outros temas), responda de forma AMIGÁVEL e EDUCADA, explicando que você não pode ajudar com esse assunto, mas que pode fornecer informações sobre:
  * Cursos oferecidos pela Faculdade de Direito (apenas Direito)
  * Processos de inscrição e acesso ao curso de Direito
  * Propinas e emolumentos do curso de Direito
  * Horários e regulamentos da Faculdade de Direito
  * Corpo docente e infraestruturas da Faculdade de Direito
  * Extensão universitária e serviços à comunidade da Faculdade de Direito
  * Qualquer outra dúvida sobre a Faculdade de Direito da Universidade Kimpa Vita

TOM DA RESPOSTA:
- Seja sempre gentil, prestativo e incentive o usuário a fazer perguntas relacionadas à Faculdade de Direito.
- NUNCA mencione cursos que não sejam de Direito (ex.: Engenharia, Ciências, Tecnologia).
- NUNCA invente informações. Se não souber, diga que não pode ajudar com esse assunto e foque no que PODE fazer.
"""


fora_do_tema_prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt_fora_do_tema),
    ("human", "{input}"),  
])

# Garanta que a chain tenha o mapeamento inicial
chain_temas_nao_relacionados = (
    RunnableParallel({
        "input": itemgetter("input")
    })
    | fora_do_tema_prompt_template
    | model_fora_do_tema
    | StrOutputParser()
)
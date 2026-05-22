from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

model_fora_do_tema = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

sys_prompt_fora_do_tema = """\
ÉS UM ASSISTENTE OFICIAL DA FACULDADE DE DIREITO DA UNIVERSIDADE KIMPA VITA.
MANTÉM UM TOM PROFISSIONAL, EDUCADO E OBJETIVO.

HISTÓRICO DA CONVERSA (apenas para coerência; NUNCA reproduzas este histórico na tua resposta):
{history}

DIRETRIZES:
- Se o utilizador te saudar, responde de forma cordial mas breve, e pergunta como podes ajudar.
- Se o utilizador perguntar algo fora do âmbito da Faculdade de Direito, responde educadamente que não podes ajudar com esse assunto e oferece os tópicos que podes abordar:
  * Cursos oferecidos pela Faculdade de Direito (apenas Direito)
  * Processos de inscrição e acesso ao curso de Direito
  * Propinas e emolumentos do curso de Direito
  * Horários e regulamentos da Faculdade de Direito
  * Corpo docente e infraestruturas (sem listar nomes que não constem da base de conhecimento)
  * Atividades de extensão e serviços à comunidade
- NUNCA inventes informação. Se não souberes, orienta o utilizador a contactar a secretaria da Faculdade.
"""

fora_do_tema_prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt_fora_do_tema),
    ("human", "{input}")
])

def formatar_history(history):
    if not history:
        return "Nenhuma conversa anterior."
    linhas = []
    for msg in history:
        if msg["role"] == "user":
            linhas.append(f"Utilizador: {msg['content']}")
        else:
            linhas.append(f"Assistente: {msg['content']}")
    return "\n".join(linhas)

chain_temas_nao_relacionados = (
    RunnableParallel({
        "input": itemgetter("input"),
        "history": itemgetter("history") | RunnableLambda(formatar_history)
    })
    | fora_do_tema_prompt_template
    | model_fora_do_tema
    | StrOutputParser()
)
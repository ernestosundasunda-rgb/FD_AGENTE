import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from db.supabase_vector import CarregadorDocumentoSupabase

load_dotenv()

model_atendimento_orientador = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

db_supabase = CarregadorDocumentoSupabase()

def supabase_retriever(query: str, k: int = 5):
    resultados = db_supabase.buscar_similares(query, k=k)
    docs = [
        Document(
            page_content=res["texto"],
            metadata={**res.get("metadata", {}), "similarity": res["similarity"], "id": res["id"]}
        )
        for res in resultados
    ]
    return docs

def formatar_contexto(documentos):
    return "\n\n".join(doc.page_content for doc in documentos)

# Prompt profissional, sem ecoar o histórico
sys_rag_prompt = """\
ÉS UM ASSISTENTE OFICIAL DA FACULDADE DE DIREITO DA UNIVERSIDADE KIMPA VITA.
MANTÉM SEMPRE UM TOM PROFISSIONAL, OBJETIVO E RESPEITOSO.

CONTEXTO DOCUMENTAL (podes usar para responder):
<contexto>
{contexto_obtido}
</contexto>

HISTÓRICO DA CONVERSA (apenas para coerência; NUNCA reproduzas este histórico na tua resposta):
{history}

REGRAS INQUEBRANTÁVEIS:
1. Responde exclusivamente com base no <contexto> fornecido.
2. Se a pergunta for uma saudação ou assunto fora da Faculdade, responde educadamente e orienta o utilizador.
3. Se o utilizador pedir uma lista (ex.: “cite‑os”) e o <contexto> NÃO contiver os nomes, responde exatamente:
   "Lamento, não tenho acesso a essa informação. Recomendo que contacte a secretaria da Faculdade de Direito."
4. NUNCA inventes informação.
5. NUNCA repitas o histórico da conversa na resposta.
6. Responde sempre em português de Angola, de forma clara e direta.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_rag_prompt),
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

chain_orientador = (
    RunnableParallel({
        "input": itemgetter("input"),
        "history": itemgetter("history") | RunnableLambda(formatar_history),
        "contexto_obtido": itemgetter("input")
                          | RunnableLambda(supabase_retriever)
                          | formatar_contexto
    })
    | prompt_template
    | model_atendimento_orientador
    | StrOutputParser()
)
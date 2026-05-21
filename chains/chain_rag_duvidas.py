import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from db.supabase_vector import CarregadorDocumentoSupabase

load_dotenv()

# Modelo LLM com temperatura zero para evitar criatividade
model_atendimento_orientador = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Conexão Supabase
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

# Prompt rigoroso – só usa o contexto
sys_rag_prompt = """\
ÉS UM ASSISTENTE DA FACULDADE DE DIREITO DA UNIVERSIDADE KIMPA VITA.
REGRAS INQUEBRANTÁVEIS:

1. SÓ PODE RESPONDER USANDO A INFORMAÇÃO DENTRO DAS TAGS <contexto> ABAIXO.
2. SE O CONTEXTO ESTIVER VAZIO OU NÃO CONTIVER A RESPOSTA, DIGA EXATAMENTE E APENAS: "Não encontrei essa informação nos documentos da Faculdade."
3. NUNCA USE CONHECIMENTO PRÉVIO. NUNCA INVENTE. NUNCA ALUCINE.
4. SEJA DIRETO E CONCISO. NÃO ACRESCENTE NADA QUE NÃO ESTEJA NO CONTEXTO.
5. RESPONDA SEMPRE EM PORTUGUÊS.

CONTEXTO:
<contexto>
{contexto_obtido}
</contexto>

PERGUNTA: {input}
RESPOSTA (apenas com base no contexto acima):
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_rag_prompt),
    ("human", "{input}")
])

chain_orientador = (
    RunnableParallel({
        "input": itemgetter("input"),
        "contexto_obtido": itemgetter("input")
                          | RunnableLambda(supabase_retriever)
                          | formatar_contexto
    })
    | prompt_template
    | model_atendimento_orientador
    | StrOutputParser()
)
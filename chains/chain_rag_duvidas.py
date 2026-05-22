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

# Prompt com histórico explícito
sys_rag_prompt = """\
Você é um assistente virtual oficial da Faculdade de Direito da Universidade Kimpa Vita.

Contexto documental (base de conhecimento):
<contexto>
{contexto_obtido}
</contexto>

HISTÓRICO DA CONVERSA (use-o obrigatoriamente para dar coerência e lembrar detalhes anteriores, como nomes e assuntos já falados):
{history}

REGRAS DE CONDUTA:
1. Responda apenas questões relacionadas à Faculdade de Direito (cursos, inscrições, propinas, horários, regulamentos, etc.).
2. Se a pergunta estiver fora do âmbito da Faculdade, diga: "Desculpe, só posso responder sobre a Faculdade de Direito da Universidade Kimpa Vita."
3. Mantenha um tom profissional e respeitoso.
4. Nunca invente informação; se não souber, oriente a contactar a secretaria académica.
5. Seja direto e objetivo.
6. Se o utilizador disser "cite-os", refira-se ao último assunto mencionado no histórico.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_rag_prompt),
    ("human", "{input}")
])

def formatar_history(history):
    """Converte a lista de mensagens no formato de texto para o prompt."""
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
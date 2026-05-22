import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
from supabase.client import create_client, Client

from chains.chain_geral import chain_temas_nao_relacionados
from chains.chain_classifica import chain_de_roteamento
from chains.chain_rag_duvidas import chain_orientador

load_dotenv()

app = FastAPI()

# CORS – permite todas as origens para testes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir ficheiros estáticos (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cliente Supabase para persistência do histórico
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ---------- Modelos ----------
class PerguntaInput(BaseModel):
    pergunta: str
    session_id: str = "default"

# ---------- Gestão de histórico no Supabase ----------
def get_history(session_id: str) -> List[Dict]:
    result = supabase.table("historico_sessoes")\
        .select("role, content")\
        .eq("session_id", session_id)\
        .order("criado_em")\
        .execute()
    return result.data

def add_to_history(session_id: str, role: str, content: str):
    supabase.table("historico_sessoes").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

# ---------- Chain principal (roteamento) ----------
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnableParallel

def executa_roteamento(entrada: dict):
    pergunta = entrada["input"]
    history = entrada.get("history", [])
    try:
        rota = entrada["resposta_pydantic"].opcao
    except Exception as e:
        print(f"⚠️ Erro na classificação: {e}. A enviar para chain geral.")
        return chain_temas_nao_relacionados.invoke({"input": pergunta, "history": history})

    if rota == 1:
        return chain_orientador.invoke({"input": pergunta, "history": history})
    else:
        return chain_temas_nao_relacionados.invoke({"input": pergunta, "history": history})

chain_principal = (
    RunnableParallel({
        "input": itemgetter("input"),
        "history": itemgetter("history"),
        "resposta_pydantic": chain_de_roteamento
    })
    | RunnableLambda(executa_roteamento)
)

# ---------- Endpoints ----------
@app.get("/")
def read_root():
    return {"status": "online", "message": "Assistente FD-UNIKIVI"}

@app.post("/chat")
def chat_endpoint(input: PerguntaInput):
    pergunta = input.pergunta
    session_id = input.session_id

    # Guardar a pergunta do utilizador no Supabase
    add_to_history(session_id, "user", pergunta)

    # Recuperar o histórico mais recente
    history = get_history(session_id)

    # LOG DE DEPURAÇÃO
    print(f"\n=== Sessão: {session_id} ===")
    print(f"Histórico ({len(history)} mensagens):")
    for msg in history:
        print(f"  [{msg['role']}] {msg['content'][:80]}...")
    print(f"Nova pergunta: {pergunta}")

    entrada = {
        "input": pergunta,
        "history": history
    }

    try:
        resposta = chain_principal.invoke(entrada)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Guardar a resposta do assistente no Supabase
    add_to_history(session_id, "assistant", resposta)

    return {"resposta": resposta, "session_id": session_id}
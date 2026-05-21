import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from operator import itemgetter

from chains.chain_rag_duvidas import chain_orientador
from chains.chain_geral import chain_temas_nao_relacionados
from chains.chain_classifica import chain_de_roteamento

load_dotenv()

# ------------------------------
# Inicialização do FastAPI
# ------------------------------
app = FastAPI(
    title="Assistente FD-UNIKIVI",
    description="API do assistente virtual da Faculdade de Direito da Universidade Kimpa Vita",
    version="1.0.0"
)

# Configuração CORS para permitir acesso de qualquer origem (ajuste em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar pasta de arquivos estáticos (opcional, para o cliente web)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # ignora se a pasta não existir

# ------------------------------
# Chains (reaproveitadas)
# ------------------------------
def executa_roteamento(entrada: dict):
    pergunta = entrada["input"]
    try:
        rota = entrada["resposta_pydantic"].opcao
    except Exception:
        # Se a classificação falhar, assume que é fora do tema
        return chain_temas_nao_relacionados.invoke({"input": pergunta})

    if rota == 1:
        return chain_orientador.invoke({"input": pergunta})
    else:
        return chain_temas_nao_relacionados.invoke({"input": pergunta})

chain_principal = (
    RunnableParallel({
        "input": itemgetter("input"),
        "resposta_pydantic": chain_de_roteamento
    })
    | RunnableLambda(executa_roteamento)
)

# ------------------------------
# Modelos de requisição/resposta
# ------------------------------
class Pergunta(BaseModel):
    pergunta: str

class Resposta(BaseModel):
    resposta: str

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"status": "online", "message": "Assistente FD-UNIKIVI"}

@app.post("/chat", response_model=Resposta)
def chat_endpoint(pergunta: Pergunta):
    try:
        resposta = chain_principal.invoke({"input": pergunta.pergunta})
        return Resposta(resposta=resposta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Execução
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
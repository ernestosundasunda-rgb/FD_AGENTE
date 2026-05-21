import sys
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from operator import itemgetter

from chains.chain_rag_duvidas import chain_orientador
from chains.chain_geral import chain_temas_nao_relacionados
from chains.chain_classifica import chain_de_roteamento

load_dotenv()

def executa_roteamento(entrada: dict):
    pergunta = entrada["input"]
    rota = entrada["resposta_pydantic"].opcao
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

if __name__ == "__main__":
    
    pergunta =  input("PERGUNTA SOBRE O INSTITUTO: ").strip()
    resposta = chain_principal.invoke({"input": pergunta})
    print(resposta)
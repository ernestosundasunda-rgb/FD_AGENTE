# query.py
import sys
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from operator import itemgetter

from chains.chain_rag_duvidas import chain_orientador
from chains.chain_geral import chain_temas_nao_relacionados
from chains.chain_classifica import chain_de_roteamento

load_dotenv()


def executa_roteamento(entrada: dict):
    """Recebe a classificação e encaminha para a chain correta."""
    pergunta = entrada["input"]
    rota = entrada["resposta_pydantic"].opcao

    if rota == 1:
        # Pergunta sobre a Faculdade de Direito → usa RAG
        return chain_orientador.invoke({"input": pergunta})
    else:
        # Saudação ou tema fora do contexto → usa chain geral
        return chain_temas_nao_relacionados.invoke({"input": pergunta})


# Chain principal que classifica e roteia
chain_principal = (
    RunnableParallel({
        "input": itemgetter("input"),
        "resposta_pydantic": chain_de_roteamento
    })
    | RunnableLambda(executa_roteamento)
)


if __name__ == "__main__":
    print("Chat iniciado. Escreva 'sair' para terminar.\n")

    while True:
        pergunta = input("Pergunta: ").strip()
        if not pergunta:
            print("Por favor, escreva uma pergunta ou 'sair' para terminar.")
            continue

        if pergunta.lower() in ("sair", "exit", "quit", "fechar"):
            print("A encerrar o chat. Até breve!")
            break

        resposta = chain_principal.invoke({"input": pergunta})
        print(resposta)
        print()   # linha em branco para separar as respostas

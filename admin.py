import sys
from db.supabase_vector import CarregadorDocumentoSupabase

def indexar():
    path = input("Caminho do PDF: ").strip()
    name = input("Nome do documento: ").strip()
    indexador = CarregadorDocumentoSupabase(path)
    indexador.indexar_informacao(name)

def listar():
    db = CarregadorDocumentoSupabase()
    docs = db.listar_documentos()
    if not docs:
        print("Nenhum documento indexado.")
    else:
        for d in docs:
            print(f"{d['documento_origem']} – {d['count']} chunks")

def remover():
    name = input("Nome exato do documento a remover: ").strip()
    db = CarregadorDocumentoSupabase()
    db.remover_documento(name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python admin.py [index|list|remove]")
        sys.exit(1)

    action = sys.argv[1]
    if action == "index":
        indexar()
    elif action == "list":
        listar()
    elif action == "remove":
        remover()
    else:
        print(f"Ação desconhecida: {action}")
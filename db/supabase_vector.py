import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase.client import create_client, Client

load_dotenv()

class CarregadorDocumentoSupabase:
    def __init__(self, caminho_documento: str = None):
        self.caminho_documento = caminho_documento
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL e SUPABASE_SERVICE_KEY devem estar definidas no .env")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200
        )

        # Modelo Gemini via API (768 dimensões)
        modelo_base = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_document"
        )
        self.modelo_embedding = modelo_base
        self.tabela_rag = "rag_documentos"

    def carregar_texto(self):
        loader = PyPDFLoader(self.caminho_documento)
        return loader.load()

    def cria_chunks(self, documentos):
        return self.text_splitter.split_documents(documentos)

    def indexar_informacao(self, nome_documento: str):
        if not self.caminho_documento:
            raise ValueError("Caminho do documento não definido.")
        paginas = self.carregar_texto()
        chunks = self.cria_chunks(paginas)
        for i, chunk in enumerate(chunks):
            embedding = self.modelo_embedding.embed_query(chunk.page_content)
            data = {
                "texto": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding,
                "chunk_id": i,
                "documento_origem": nome_documento
            }
            self.supabase.table(self.tabela_rag).insert(data).execute()
        print(f"Indexação concluída: '{nome_documento}' com {len(chunks)} chunks.")

    def listar_documentos(self):
        res = self.supabase.table(self.tabela_rag).select("documento_origem").execute()
        docs = {}
        for r in res.data:
            docs[r['documento_origem']] = docs.get(r['documento_origem'], 0) + 1
        return [{"documento_origem": k, "count": v} for k, v in docs.items()]

    def remover_documento(self, nome_documento: str):
        self.supabase.table(self.tabela_rag)\
            .delete()\
            .eq("documento_origem", nome_documento)\
            .execute()
        print(f"Documento '{nome_documento}' removido.")

    def buscar_similares(self, consulta: str, k: int = 5) -> List[Dict]:
        query_emb = self.modelo_embedding.embed_query(consulta)
        result = self.supabase.rpc(
            "match_rag_documentos",
            {
                "query_embedding": query_emb,
                "match_threshold": 0.4,
                "match_count": k
            }
        ).execute()
        return result.data
from db.supabase_vector import CarregadorDocumentoSupabase

db = CarregadorDocumentoSupabase()

# 1. Listar documentos indexados
print("📚 Documentos indexados:")
docs = db.listar_documentos()
for d in docs:
    print(f"   - {d['documento_origem']} ({d['count']} chunks)")

# 2. Testar perguntas específicas
perguntas = [
    "Quando tiveram início as actividades da Faculdade de Direito?",
    "Onde funcionava inicialmente a Faculdade de Direito?",
    "Em que região académica a UNIKIVI está inserida?",
    "Qual é a missão da Faculdade de Direito?"
]

for pergunta in perguntas:
    print(f"\n❓ Pergunta: {pergunta}")
    resultados = db.buscar_similares(pergunta, k=5)
    if not resultados:
        print("   ⚠️ Nenhum chunk encontrado!")
    else:
        for i, res in enumerate(resultados):
            print(f"   [{i+1}] (sim: {res['similarity']:.3f}) {res['texto'][:120]}...")
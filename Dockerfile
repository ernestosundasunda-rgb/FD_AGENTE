# Usar uma imagem base leve do Python
FROM python:3.11-slim

# Definir o diretório de trabalho dentro do contentor
WORKDIR /app

# Copiar o ficheiro de dependências primeiro (para aproveitar a cache do Docker)
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto do código da aplicação
COPY . .

# Expor a porta que o Render vai usar (normalmente 8000, mas o Render define a variável PORT)
EXPOSE 8000

# Comando para iniciar a API com uvicorn
# O Render define automaticamente a variável PORT, por isso usamos $PORT com fallback para 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
# Dockerfile
# Usar uma imagem base oficial do Python
FROM python:3.10-slim

# Definir o diretório de trabalho no container
WORKDIR /app

# Instalar dependências do sistema necessárias para matplotlib
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar o arquivo de requirements primeiro para aproveitamento do cache do Docker
COPY requirements.txt .

# Instalar as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto dos arquivos do projeto
COPY . .

# Expor a porta que a aplicação vai rodar
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
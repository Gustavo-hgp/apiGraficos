# JULIOS - Controle Financeiro Pessoal Inteligente 💰📊

## 🌟 Sobre o Projeto

JULIUS é uma aplicação inovadora de geração de gráficos financeiros, projetada para ajudar você a visualizar e entender melhor seus gastos e receitas de forma simples e intuitiva.

### 🎯 Funcionalidades Principais

- Geração de gráficos de pizza para distribuição de despesas
- Criação de gráficos de barras para visualização de receitas
- API securizada com autenticação por chave
- Suporte a múltiplas categorias financeiras
- Personalização de títulos e paletas de cores

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.10+
- pip (Gerenciador de pacotes Python)

### Passos de Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/cofipei.git
cd cofipei
```

2. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Inicie o servidor:
```bash
uvicorn main:app --reload
```

5. Acesse a documentação da API:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔐 Autenticação

Para usar a API, utilize a chave de API no cabeçalho:
```
X-API-Key: cofipei_2024_secret_key_financas
```

## 📋 Exemplo de Uso

### Payload de Exemplo

```json
{
    "pizza_chart": {
        "Alimentação": 850.50,
        "Combustível": 450.75,
        "Educação": 350.00,
        "Lazer": 250.25,
        "Transporte": 200.00,
        "Saúde": 300.50
    },
    "bar_chart": {
        "Salário": 5000.00,
        "Freelance": 1200.00,
        "Investimentos": 750.50,
        "Outros Rendimentos": 350.75
    },
    "pizza_title": "Distribuição de Despesas",
    "bar_title": "Fontes de Receita"
}
```

### Exemplo de cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/generate-chart' \
  -H 'X-API-Key: cofipei_2024_secret_key_financas' \
  -H 'Content-Type: application/json' \
  -d '{JSON_PAYLOAD_AQUI}'
```

## 🐳 Dockerização

1. Construa a imagem:
```bash
docker build -t cofipei .
```

2. Execute o container:
```bash
docker run -p 8000:8000 cofipei
```

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de enviar um pull request.

## 🙏 Agradecimentos Especiais

### Comunidade de Desenvolvedores
- A todos os desenvolvedores que compartilham conhecimento gratuitamente
- Comunidades de código aberto que tornam projetos como este possíveis

### Tecnologias Utilizadas
- FastAPI
- Matplotlib
- Python
- Docker

### Comunidades de Apoio
- Grupos de desenvolvedores Python
- Comunidades de finanças pessoais
- Fóruns de tecnologia e programação

## 📜 Licença

[Inserir detalhes da licença]

---

⭐ Deixe uma estrela se este projeto te ajudou! 

Desenvolvido com ❤️ pela Comunidade COFIPEI
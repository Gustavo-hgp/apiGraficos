"""
Julius - API para Geração de Gráficos Financeiros
Desenvolvido com FastAPI e Matplotlib (sem banco de dados)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO
from decimal import Decimal

# Configuração do matplotlib para melhor aparência
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

app = FastAPI(
    title="Julius - API de Gráficos Financeiros",
    description="API para geração de relatórios gráficos de receitas e despesas",
    version="1.0.0"
)

# =====================================================
# MODELOS PYDANTIC
# =====================================================

class LancamentoModel(BaseModel):
    data: date = Field(..., description="Data do lançamento")
    descricao: str = Field(..., description="Descrição do lançamento")
    valor: float = Field(..., gt=0, description="Valor do lançamento (sempre positivo)")
    categoria: str = Field(..., description="Categoria do lançamento")
    tipo: str = Field(..., description="Tipo: Receita ou Despesa")

class PeriodoModel(BaseModel):
    data_inicial: date
    data_final: date

class RelatorioResponse(BaseModel):
    periodo: PeriodoModel
    total_despesas: float
    total_receitas: float
    imagem: str = Field(..., description="Gráfico codificado em base64")

class GraficoRequest(BaseModel):
    lancamentos: List[LancamentoModel] = Field(..., description="Lista de lançamentos financeiros")
    tipo_grafico: str = Field("pizza", description="Tipo: pizza, barra, linha")
    titulo: Optional[str] = Field(None, description="Título personalizado do gráfico")

# =====================================================
# FUNÇÕES UTILITÁRIAS
# =====================================================

def processar_dados(lancamentos: List[LancamentoModel]) -> Dict:
    """Processa os dados dos lançamentos"""
    if not lancamentos:
        return {"dados": [], "total_receitas": 0, "total_despesas": 0, "periodo": None}
    
    # Converter para DataFrame para facilitar manipulação
    dados_dict = []
    for lanc in lancamentos:
        dados_dict.append({
            'data': lanc.data,
            'descricao': lanc.descricao,
            'valor': lanc.valor,
            'categoria': lanc.categoria,
            'tipo': lanc.tipo
        })
    
    df = pd.DataFrame(dados_dict)
    
    # Calcular totais
    total_receitas = df[df['tipo'] == 'Receita']['valor'].sum()
    total_despesas = df[df['tipo'] == 'Despesa']['valor'].sum()
    
    # Agrupar por categoria e tipo
    dados_agrupados = df.groupby(['categoria', 'tipo'])['valor'].sum().reset_index()
    dados_agrupados = dados_agrupados.to_dict('records')
    
    # Calcular período
    data_inicial = df['data'].min()
    data_final = df['data'].max()
    
    return {
        "dados": dados_agrupados,
        "dados_temporais": dados_dict,
        "total_receitas": float(total_receitas),
        "total_despesas": float(total_despesas),
        "periodo": {"data_inicial": data_inicial, "data_final": data_final}
    }

def calcular_totais_por_categoria(dados: List[Dict]) -> Dict:
    """Calcula totais agrupados por categoria"""
    receitas = {}
    despesas = {}
    
    for item in dados:
        categoria = item['categoria']
        valor = item['valor']
        tipo = item['tipo']
        
        if tipo == 'Receita':
            receitas[categoria] = receitas.get(categoria, 0) + valor
        else:
            despesas[categoria] = despesas.get(categoria, 0) + valor
    
    return {"receitas": receitas, "despesas": despesas}

# =====================================================
# FUNÇÕES DE GERAÇÃO DE GRÁFICOS
# =====================================================

def gerar_grafico_pizza(dados: List[Dict], titulo: str = "Distribuição por Categoria"):
    """Gera gráfico de pizza das categorias"""
    totais = calcular_totais_por_categoria(dados)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico de Receitas
    if totais['receitas']:
        labels_r = list(totais['receitas'].keys())
        valores_r = list(totais['receitas'].values())
        
        colors_r = plt.cm.Greens([0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4])[:len(valores_r)]
        wedges, texts, autotexts = ax1.pie(valores_r, labels=labels_r, autopct='%1.1f%%', 
                                          startangle=90, colors=colors_r)
        ax1.set_title('📈 Receitas por Categoria', fontsize=16, fontweight='bold', pad=20)
        
        # Melhorar aparência dos textos
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax1.text(0.5, 0.5, '❌ Sem dados de receitas', ha='center', va='center', 
                fontsize=14, transform=ax1.transAxes)
        ax1.set_title('📈 Receitas por Categoria', fontsize=16, fontweight='bold', pad=20)
    
    # Gráfico de Despesas
    if totais['despesas']:
        labels_d = list(totais['despesas'].keys())
        valores_d = list(totais['despesas'].values())
        
        colors_d = plt.cm.Reds([0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4])[:len(valores_d)]
        wedges, texts, autotexts = ax2.pie(valores_d, labels=labels_d, autopct='%1.1f%%', 
                                          startangle=90, colors=colors_d)
        ax2.set_title('📉 Despesas por Categoria', fontsize=16, fontweight='bold', pad=20)
        
        # Melhorar aparência dos textos
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, '✅ Sem despesas', ha='center', va='center', 
                fontsize=14, transform=ax2.transAxes)
        ax2.set_title('📉 Despesas por Categoria', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle(titulo, fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    return fig

def gerar_grafico_barras(dados: List[Dict], titulo: str = "Receitas vs Despesas por Categoria"):
    """Gera gráfico de barras comparativo"""
    if not dados:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, '❌ Sem dados para exibir', ha='center', va='center', 
               fontsize=16, transform=ax.transAxes)
        ax.set_title(titulo, fontsize=16, fontweight='bold')
        return fig
    
    # Preparar dados
    df = pd.DataFrame(dados)
    pivot_df = df.pivot_table(values='valor', index='categoria', columns='tipo', fill_value=0)
    
    # Se não há colunas de Receita ou Despesa, adicionar com zero
    if 'Receita' not in pivot_df.columns:
        pivot_df['Receita'] = 0
    if 'Despesa' not in pivot_df.columns:
        pivot_df['Despesa'] = 0
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = range(len(pivot_df.index))
    width = 0.35
    
    receitas = pivot_df['Receita'].values
    despesas = pivot_df['Despesa'].values
    
    bars1 = ax.bar([i - width/2 for i in x], receitas, width, 
                   label='💰 Receitas', color='#4ecdc4', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], despesas, width, 
                   label='💸 Despesas', color='#ff6b6b', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'R$ {height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'R$ {height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('📂 Categorias', fontsize=14, fontweight='bold')
    ax.set_ylabel('💵 Valores (R$)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Formatação dos valores no eixo y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
    
    plt.tight_layout()
    return fig

def gerar_grafico_linha_tempo(dados_temporais: List[Dict], titulo: str = "Fluxo de Caixa Temporal"):
    """Gera gráfico de linha temporal do fluxo de caixa"""
    if not dados_temporais:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, '❌ Sem dados para o período', ha='center', va='center', 
               fontsize=16, transform=ax.transAxes)
        ax.set_title(titulo, fontsize=16, fontweight='bold')
        return fig
    
    # Preparar DataFrame
    df_tempo = pd.DataFrame(dados_temporais)
    df_tempo['data'] = pd.to_datetime(df_tempo['data'])
    
    # Agrupar por data e tipo
    df_agrupado = df_tempo.groupby(['data', 'tipo'])['valor'].sum().reset_index()
    pivot_tempo = df_agrupado.pivot_table(values='valor', index='data', columns='tipo', fill_value=0)
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plotar linhas
    if 'Receita' in pivot_tempo.columns:
        ax.plot(pivot_tempo.index, pivot_tempo['Receita'], 
               marker='o', linewidth=3, label='💰 Receitas', color='#4ecdc4', markersize=8)
    
    if 'Despesa' in pivot_tempo.columns:
        ax.plot(pivot_tempo.index, pivot_tempo['Despesa'], 
               marker='s', linewidth=3, label='💸 Despesas', color='#ff6b6b', markersize=8)
    
    # Calcular e plotar saldo acumulado
    receitas_cum = pivot_tempo.get('Receita', pd.Series(0, index=pivot_tempo.index)).cumsum()
    despesas_cum = pivot_tempo.get('Despesa', pd.Series(0, index=pivot_tempo.index)).cumsum()
    saldo_acum = receitas_cum - despesas_cum
    
    ax.plot(pivot_tempo.index, saldo_acum, 
           marker='^', linewidth=4, label='📊 Saldo Acumulado', 
           color='#45b7d1', alpha=0.8, markersize=10)
    
    # Adicionar linha zero para referência
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title(titulo, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('📅 Data', fontsize=14, fontweight='bold')
    ax.set_ylabel('💵 Valores (R$)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Formatação dos eixos
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x:,.0f}'))
    
    plt.tight_layout()
    return fig

def converter_figura_base64(fig):
    """Converte figura matplotlib para base64"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    
    # Converter para base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # Liberar memória
    
    return img_base64

# =====================================================
# ENDPOINTS DA API
# =====================================================

@app.get("/")
async def root():
    """Endpoint raiz da API"""
    return {
        "app": "Julius - API de Gráficos Financeiros",
        "version": "1.0.0",
        "description": "API para geração de gráficos baseada em dados de entrada",
        "endpoints": [
            "POST /relatorio/pizza",
            "POST /relatorio/barras", 
            "POST /relatorio/linha",
            "POST /relatorio/customizado"
        ]
    }

@app.post("/relatorio/pizza", response_model=RelatorioResponse)
async def gerar_relatorio_pizza(request: GraficoRequest):
    """Gera relatório com gráfico de pizza"""
    
    if not request.lancamentos:
        raise HTTPException(status_code=400, detail="Lista de lançamentos não pode estar vazia")
    
    # Processar dados
    dados_processados = processar_dados(request.lancamentos)
    
    # Título personalizado ou padrão
    titulo = request.titulo or f"Relatório Financeiro - {dados_processados['periodo']['data_inicial']} a {dados_processados['periodo']['data_final']}"
    
    # Gerar gráfico
    fig = gerar_grafico_pizza(dados_processados['dados'], titulo)
    img_base64 = converter_figura_base64(fig)
    
    return RelatorioResponse(
        periodo=PeriodoModel(
            data_inicial=dados_processados['periodo']['data_inicial'],
            data_final=dados_processados['periodo']['data_final']
        ),
        total_receitas=dados_processados['total_receitas'],
        total_despesas=dados_processados['total_despesas'],
        imagem=img_base64
    )

@app.post("/relatorio/barras", response_model=RelatorioResponse)
async def gerar_relatorio_barras(request: GraficoRequest):
    """Gera relatório com gráfico de barras"""
    
    if not request.lancamentos:
        raise HTTPException(status_code=400, detail="Lista de lançamentos não pode estar vazia")
    
    # Processar dados
    dados_processados = processar_dados(request.lancamentos)
    
    # Título personalizado ou padrão
    titulo = request.titulo or f"Comparativo por Categoria - {dados_processados['periodo']['data_inicial']} a {dados_processados['periodo']['data_final']}"
    
    # Gerar gráfico
    fig = gerar_grafico_barras(dados_processados['dados'], titulo)
    img_base64 = converter_figura_base64(fig)
    
    return RelatorioResponse(
        periodo=PeriodoModel(
            data_inicial=dados_processados['periodo']['data_inicial'],
            data_final=dados_processados['periodo']['data_final']
        ),
        total_receitas=dados_processados['total_receitas'],
        total_despesas=dados_processados['total_despesas'],
        imagem=img_base64
    )

@app.post("/relatorio/linha", response_model=RelatorioResponse)
async def gerar_relatorio_linha(request: GraficoRequest):
    """Gera relatório com gráfico de linha temporal"""
    
    if not request.lancamentos:
        raise HTTPException(status_code=400, detail="Lista de lançamentos não pode estar vazia")
    
    # Processar dados
    dados_processados = processar_dados(request.lancamentos)
    
    # Título personalizado ou padrão
    titulo = request.titulo or f"Fluxo de Caixa - {dados_processados['periodo']['data_inicial']} a {dados_processados['periodo']['data_final']}"
    
    # Gerar gráfico
    fig = gerar_grafico_linha_tempo(dados_processados['dados_temporais'], titulo)
    img_base64 = converter_figura_base64(fig)
    
    return RelatorioResponse(
        periodo=PeriodoModel(
            data_inicial=dados_processados['periodo']['data_inicial'],
            data_final=dados_processados['periodo']['data_final']
        ),
        total_receitas=dados_processados['total_receitas'],
        total_despesas=dados_processados['total_despesas'],
        imagem=img_base64
    )

@app.post("/relatorio/customizado", response_model=RelatorioResponse)
async def gerar_relatorio_customizado(request: GraficoRequest):
    """Gera relatório com tipo de gráfico customizado"""
    
    if not request.lancamentos:
        raise HTTPException(status_code=400, detail="Lista de lançamentos não pode estar vazia")
    
    # Processar dados
    dados_processados = processar_dados(request.lancamentos)
    
    # Gerar gráfico baseado no tipo solicitado
    if request.tipo_grafico == "pizza":
        titulo = request.titulo or "Distribuição por Categoria"
        fig = gerar_grafico_pizza(dados_processados['dados'], titulo)
    elif request.tipo_grafico == "barra":
        titulo = request.titulo or "Comparativo por Categoria"
        fig = gerar_grafico_barras(dados_processados['dados'], titulo)
    elif request.tipo_grafico == "linha":
        titulo = request.titulo or "Fluxo de Caixa Temporal"
        fig = gerar_grafico_linha_tempo(dados_processados['dados_temporais'], titulo)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Tipo de gráfico não suportado. Use: pizza, barra, linha"
        )
    
    img_base64 = converter_figura_base64(fig)
    
    return RelatorioResponse(
        periodo=PeriodoModel(
            data_inicial=dados_processados['periodo']['data_inicial'],
            data_final=dados_processados['periodo']['data_final']
        ),
        total_receitas=dados_processados['total_receitas'],
        total_despesas=dados_processados['total_despesas'],
        imagem=img_base64
    )

# =====================================================
# EXECUTAR APLICAÇÃO
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

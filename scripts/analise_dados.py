import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import requests
from sklearn.preprocessing import MinMaxScaler

# %% Configurações gerais para visualizações

# Define o estilo dos gráficos com Seaborn e o tamanho padrão das figuras
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %% 1. Carregamento dos Dados Processados

# Carrega as bases de dados processadas e sem inconsistências
transacoes = pd.read_csv('transacoes_sem_inconsistencias.csv')
propostas = pd.read_csv('propostas_credito_sem_inconsistencias.csv')
contas = pd.read_csv('contas_sem_inconsistencias.csv')
colaboradores = pd.read_csv('colaboradores_processado.csv')
clientes = pd.read_csv('clientes_processado.csv')
agencias = pd.read_csv('agencias_processado.csv')

# %% 2. Conversão e Processamento de Datas

# Cria um dicionario para mapear as colunas de data 
date_cols = {
    'transacoes': ('data_transacao', '%Y-%m'),
    'propostas': ('data_entrada_proposta', '%Y-%m'),
    'contas': [('data_abertura', '%Y-%m'), ('data_ultimo_lancamento', '%Y-%m')],
    'clientes': ('data_inclusao', '%Y-%m'),
    'agencias': ('data_abertura', '%Y-%m-%d')
}

# Converte as colunas de data para o formato datetime, permitindo operações temporais
transacoes['data_transacao'] = pd.to_datetime(transacoes['data_transacao'], format='%Y-%m', errors='coerce')
propostas['data_entrada_proposta'] = pd.to_datetime(propostas['data_entrada_proposta'], format='%Y-%m', errors='coerce')
contas['data_abertura'] = pd.to_datetime(contas['data_abertura'], format='%Y-%m', errors='coerce')
contas['data_ultimo_lancamento'] = pd.to_datetime(contas['data_ultimo_lancamento'], format='%Y-%m', errors='coerce')
clientes['data_inclusao'] = pd.to_datetime(clientes['data_inclusao'], format='%Y-%m', errors='coerce')
agencias['data_abertura'] = pd.to_datetime(agencias['data_abertura'], format='%Y-%m-%d', errors='coerce')

# %% 3. Análises Exploratórias e Estatísticas

# %% 3.1 Análise das Transações

# Agrupa as transações mensalmente, calculando o número e o volume total
transacoes_monthly = transacoes.groupby(pd.Grouper(key='data_transacao', freq='M')).agg(
    num_transacoes=('valor_transacao_abs', 'count'),
    volume_total=('valor_transacao_abs', 'sum')
).reset_index()

# Gráficos em subplots para o número de transações e o volume total
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
sns.lineplot(data=transacoes_monthly, x='data_transacao', y='num_transacoes', marker='o', ax=axes[0])
axes[0].set_title('Número de Transações Mensais')
axes[0].set_xlabel('Data')
axes[0].set_ylabel('Transações')
axes[0].tick_params(axis='x', rotation=45)

sns.lineplot(data=transacoes_monthly, x='data_transacao', y='volume_total', marker='o', color='green', ax=axes[1])
axes[1].set_title('Volume Total Mensal de Transações')
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Volume Total')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% 3.2 Análise das Propostas de Crédito

# Agrupa as propostas por data de entrada e status para avaliar a evolução dos pedidos
propostas_agg = propostas.groupby([pd.Grouper(key='data_entrada_proposta', freq='M'), 'status_proposta']).agg(
    num_propostas=('cod_proposta', 'count')
).reset_index()

# Cria subplots (2x2) para cada status de proposta
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()
for i, status in enumerate(propostas_agg['status_proposta'].unique()):
    dados = propostas_agg[propostas_agg['status_proposta'] == status]
    sns.lineplot(data=dados, x='data_entrada_proposta', y='num_propostas', marker='o', ax=axes[i])
    axes[i].set_title(f"Status: {status}")
    axes[i].set_xlabel("Data de Entrada")
    axes[i].set_ylabel("Propostas")
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle('Propostas de Crédito por Status')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Boxplot que mostra a distribuição dos valores propostos para cada status 
sns.boxplot(x='status_proposta', y='valor_proposta', data=propostas)
plt.title('Valor Proposto por Status')
plt.xlabel('Status')
plt.ylabel('Valor Proposto')
plt.tight_layout()
plt.show()

# Histograma (com curva KDE) da quantidade de parcelas das propostas
sns.histplot(propostas['quantidade_parcelas'], bins=30, kde=True)
plt.title('Quantidade de Parcelas')
plt.xlabel('Parcelas')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# Histograma (com curva KDE) do período de carência das propostas
sns.histplot(propostas['carencia'], bins=30, kde=True)
plt.title('Período de Carência')
plt.xlabel('Meses')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# Cálculo da taxa de aprovação das propostas de crédito
total_propostas = propostas['cod_proposta'].count()
propostas_aprovadas = propostas[propostas['status_proposta'].str.lower() == 'aprovada']['cod_proposta'].count()
taxa_aprovacao = (propostas_aprovadas / total_propostas) * 100
print(f"Taxa de Aprovação de Propostas de Crédito: {taxa_aprovacao:.2f}%\n")

# %% 3.3 Análise das Contas

# Distribuição dos saldos (disponível e total) e a correlação entre eles
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
sns.histplot(contas['saldo_disponivel'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Saldo Disponível')
axes[0].set_xlabel('Saldo Disponível')
axes[0].set_ylabel('Frequência')

sns.histplot(contas['saldo_total'], bins=30, kde=True, ax=axes[1])
axes[1].set_title('Saldo Total')
axes[1].set_xlabel('Saldo Total')
axes[1].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

# Relação entre o saldo total e o saldo disponível
sns.scatterplot(x='saldo_total', y='saldo_disponivel', data=contas, alpha=0.7)
plt.title('Saldo Total vs. Saldo Disponível')
plt.xlabel('Saldo Total')
plt.ylabel('Saldo Disponível')
plt.tight_layout()
plt.show()

# Imprime a correlação entre o saldo total e o saldo disponível
print(f"Correlação entre Saldo Total e Saldo Disponível: {contas['saldo_total'].corr(contas['saldo_disponivel']):.4f}\n")

# Quantidade das contas abertas
contas_agg_data = contas.groupby(pd.Grouper(key='data_abertura', freq='M')).agg(num_contas=('num_conta', 'count')).reset_index()
sns.lineplot(data=contas_agg_data, x='data_abertura', y='num_contas', marker='o')
plt.title('Contas Abertas por Mês')
plt.xlabel('Data')
plt.ylabel('Contas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crescimento no número de contas
contas_agg_data['num_contas_acumuladas'] = contas_agg_data['num_contas'].cumsum()
sns.lineplot(data=contas_agg_data, x='data_abertura', y='num_contas_acumuladas', marker='o')
plt.title('Crescimento no número de contas')
plt.xlabel('Data')
plt.ylabel('Contas Acumuladas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Graficos de barras: Agências por UF e de Contas por UF
agencias_por_uf = agencias.groupby('uf').agg(total_agencias=('cod_agencia', 'count')).reset_index()
contas_com_uf = contas.merge(agencias[['cod_agencia', 'uf']], on='cod_agencia', how='left')
contas_por_uf = contas_com_uf.groupby('uf').agg(total_contas=('num_conta', 'count')).reset_index()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
sns.barplot(data=agencias_por_uf, x='uf', y='total_agencias', palette='viridis', ax=axes[0])
axes[0].set_title('Agências por UF')
axes[0].set_xlabel('UF')
axes[0].set_ylabel('Total de Agências')

sns.barplot(data=contas_por_uf, x='uf', y='total_contas', palette='Blues_d', ax=axes[1])
axes[1].set_title('Contas por UF')
axes[1].set_xlabel('UF')
axes[1].set_ylabel('Total de Contas')

plt.tight_layout()
plt.show()

# %% 3.4 Análise dos Colaboradores

# Número de colaboradores por agência
colab_by_agencia = colaboradores.groupby('cod_agencia').size().reset_index(name='num_colaboradores')
sns.barplot(data=colab_by_agencia, x='cod_agencia', y='num_colaboradores', palette='viridis')
plt.title('Número de Colaboradores por Agência')
plt.xlabel('Agência')
plt.ylabel('Número de Colaboradores')
plt.figtext(0.5, 0.01, "Comentário: Este gráfico mostra o número de colaboradores por agência.", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.show()

# Distribuição das idades dos colaboradores.
sns.histplot(data=colaboradores, x='idade', bins=20, kde=True)
plt.title('Distribuição de Idade dos Colaboradores')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.figtext(0.5, 0.01, "Comentário: Este histograma exibe a distribuição das idades dos colaboradores.", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.show()

# %% 3.5 Análise dos Clientes

# Distribuição das idades dos clientes.
sns.histplot(clientes['idade'], bins=20, kde=True)
plt.title('Distribuição de Idade dos Clientes')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# %% 3.6 Integração: Clientes e Transações

# Realiza a integração dos dados de clientes e transações
# Renomeia a coluna 'num_conta' para 'cod_cliente' e realiza o merge
transacoes.rename(columns={'num_conta': 'cod_cliente'}, inplace=True)
transacoes_clientes = transacoes.merge(clientes[['cod_cliente', 'idade']], on='cod_cliente', how='left')

# Agrega os dados por cliente para calcular o volume total, a média e a quantidade de transações
cliente_transacoes = transacoes_clientes.groupby('cod_cliente').agg(
    total_volume=('valor_transacao_abs', 'sum'),
    media_volume=('valor_transacao_abs', 'mean'),
    total_transactions=('valor_transacao_abs', 'count')
).reset_index()
cliente_transacoes = cliente_transacoes.merge(clientes[['cod_cliente', 'idade']], on='cod_cliente', how='left')

# Cria faixas etárias para analisar a relação entre idade, volume e quantidade de transações
bins = [0, 20, 30, 40, 50, 60, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']
cliente_transacoes['faixa_idade'] = pd.cut(cliente_transacoes['idade'], bins=bins, labels=labels, right=False)

# Cria uma coluna com o valor médio de cada faixa para análise de regressão
mid_points = {'0-20': 10, '21-30': 25.5, '31-40': 35.5, '41-50': 45.5, '51-60': 55.5, '61+': 70.5}
cliente_transacoes['idade_mid'] = cliente_transacoes['faixa_idade'].map(mid_points)

# Agrega os dados por faixa etária, calculando a média do volume e a soma total de transações
agg_faixa = cliente_transacoes.groupby(['faixa_idade', 'idade_mid']).agg(
    total_volume_mean=('total_volume', 'mean'),
    total_transactions_sum=('total_transactions', 'sum')
).reset_index()

# Gráficos de barras para análise por faixa etária: Média do Volume Total por Faixa de Idade e Total de Transações por Faixa de Idade
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

sns.barplot(x='faixa_idade', y='total_volume_mean', data=agg_faixa, ax=axes[0], palette='viridis')
axes[0].set_title('Média do Volume Total por Faixa de Idade')
axes[0].set_xlabel('Faixa de Idade')
axes[0].set_ylabel('Volume Total Médio')

sns.barplot(x='faixa_idade', y='total_transactions_sum', data=agg_faixa, ax=axes[1], palette='magma')
axes[1].set_title('Total de Transações por Faixa de Idade')
axes[1].set_xlabel('Faixa de Idade')
axes[1].set_ylabel('Total de Transações')

plt.figtext(0.5, 0.01, 
            "Comentário: O gráfico superior mostra a média do volume total por faixa etária, enquanto o gráfico inferior exibe o total de transações por faixa etária.", 
            ha="center", fontsize=10, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# Gráficos de regressão para análise por faixa etária: Idade Média e Volume Total Médio por Faixa de Idade e Idade Média e Total de Transações por Faixa de Idade
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

sns.regplot(x='idade_mid', y='total_volume_mean', data=agg_faixa, ax=axes[0], marker='o', color='green')
axes[0].set_title('Regressão: Idade Média vs. Volume Total Médio por Faixa de Idade')
axes[0].set_xlabel('Idade Média da Faixa')
axes[0].set_ylabel('Volume Total Médio')

sns.regplot(x='idade_mid', y='total_transactions_sum', data=agg_faixa, ax=axes[1], marker='o', color='blue')
axes[1].set_title('Regressão: Idade Média vs. Total de Transações por Faixa de Idade')
axes[1].set_xlabel('Idade Média da Faixa')
axes[1].set_ylabel('Total de Transações')

plt.figtext(0.5, 0.01, 
            "Comentário: O gráfico superior de regressão mostra a relação entre a idade média e o volume total médio, enquanto o gráfico inferior exibe a relação entre a idade média e o total de transações por faixa etária.", 
            ha="center", fontsize=10, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# %% 3.7 Análise das Agências

#Integra os dados de contas com os dados de agências para análise comparativa
contas_com_agencia = contas.merge(agencias[['cod_agencia', 'nome', 'cidade', 'uf', 'tipo_agencia']], 
                                  on='cod_agencia', how='left')
contas_por_tipo = contas_com_agencia.groupby('tipo_agencia').agg(cod_cliente=('cod_cliente', 'count')).reset_index()
print("Número de contas por tipo de agência:\n", contas_por_tipo, "\n")

transacoes_com_agencia = transacoes.merge(contas_com_agencia[['cod_cliente', 'tipo_agencia']], 
                                          on='cod_cliente', how='left')
transacoes_por_tipo = transacoes_com_agencia.groupby('tipo_agencia').agg(
    num_transacoes=('valor_transacao_abs', 'count'),
    volume_transacoes=('valor_transacao_abs', 'sum')
).reset_index()
print("Transações por tipo de agência:\n", transacoes_por_tipo, "\n")

# Cria um grid 2x2 para exibir os gráficos relacionados às agências
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# Contagem de Agências por Tipo
sns.countplot(x='tipo_agencia', data=agencias, ax=axes[0,0], palette='Set2')
axes[0,0].set_title('Contagem de Agências por Tipo')
axes[0,0].set_xlabel('Tipo de Agência')
axes[0,0].set_ylabel('Contagem')

# Número de Contas por Tipo de Agência
sns.barplot(data=contas_por_tipo, x='tipo_agencia', y='cod_cliente', ax=axes[0,1], palette='Blues_d')
axes[0,1].set_title('Número de Contas por Tipo de Agência')
axes[0,1].set_xlabel('Tipo de Agência')
axes[0,1].set_ylabel('Número de Contas')

# Número de Transações por Tipo de Agência
sns.barplot(data=transacoes_por_tipo, x='tipo_agencia', y='num_transacoes', ax=axes[1,0], palette='Greens_d')
axes[1,0].set_title('Número de Transações por Tipo de Agência')
axes[1,0].set_xlabel('Tipo de Agência')
axes[1,0].set_ylabel('Número de Transações')

# Volume de Transações por Tipo de Agência
sns.barplot(data=transacoes_por_tipo, x='tipo_agencia', y='volume_transacoes', ax=axes[1,1], palette='Oranges_d')
axes[1,1].set_title('Volume de Transações por Tipo de Agência')
axes[1,1].set_xlabel('Tipo de Agência')
axes[1,1].set_ylabel('Volume de Transações')

plt.tight_layout()
plt.show()

# %% 4. Dimensão de Datas e Análises Temporais

# Cria uma dimensão de datas para análises temporais detalhadas
dates = pd.date_range(start=transacoes['data_transacao'].min(), end=transacoes['data_transacao'].max(), freq='D')
dim_dates = pd.DataFrame({'date': dates})
dim_dates['year'] = dim_dates['date'].dt.year
dim_dates['month'] = dim_dates['date'].dt.month
dim_dates['month_name'] = dim_dates['date'].dt.month_name(locale='pt_BR')
dim_dates['quarter'] = dim_dates['date'].dt.quarter
print("Dimensão de Datas (exemplo):\n", dim_dates.head(), "\n")

# %% 4.1 Transações por Trimestre

# Calcula a média de transações e volume médio
transacoes['quarter'] = transacoes['data_transacao'].dt.quarter
transacoes_quarter = transacoes.groupby('quarter').agg(
    media_transacoes=('valor_transacao_abs', 'count'),
    media_volume=('valor_transacao_abs', 'mean')
).reset_index()

# Gráficos de barras para transações e volume por trimestre
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
sns.barplot(x='quarter', y='media_transacoes', data=transacoes_quarter, palette='Blues_d', ax=axes[0])
axes[0].set_title('Transações por Trimestre (Média)')
axes[0].set_xlabel('Trimestre')
axes[0].set_ylabel('Transações')

sns.barplot(x='quarter', y='media_volume', data=transacoes_quarter, palette='Greens_d', ax=axes[1])
axes[1].set_title('Volume Médio por Trimestre')
axes[1].set_xlabel('Trimestre')
axes[1].set_ylabel('Volume Médio')

plt.tight_layout()
plt.show()

# %% 4.2 Meses com "r" no Nome

# Calcula a média de transações e volume médio
transacoes['month_name'] = transacoes['data_transacao'].dt.month_name(locale='pt_BR').str.lower()
transacoes['month_has_r'] = transacoes['month_name'].apply(lambda x: 'r' in x)
transacoes_by_month = transacoes.groupby('month_has_r').agg(
    avg_transacoes=('valor_transacao_abs', 'count'),
    avg_volume=('valor_transacao_abs', 'mean')
).reset_index()

# Gráficos de barras para transações e volume nos meses com e sem "r"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
sns.barplot(x='month_has_r', y='avg_transacoes', data=transacoes_by_month, palette='Purples_d', ax=axes[0])
axes[0].set_title('Transações: Meses com/sem "r"')
axes[0].set_xlabel('Mês contém "r"')
axes[0].set_ylabel('Transações')

sns.barplot(x='month_has_r', y='avg_volume', data=transacoes_by_month, palette='Oranges_d', ax=axes[1])
axes[1].set_title('Volume: Meses com/sem "r"')
axes[1].set_xlabel('Mês contém "r"')
axes[1].set_ylabel('Volume Médio')

plt.tight_layout()
plt.show()

# %% 5. Integração de Dados Externos

# %% 5.1 Preparar coluna para agregação mensal

# Converte a data das transações para períodos mensais e agrupa as transações
transacoes['year_month'] = transacoes['data_transacao'].dt.to_period('M').astype(str)
transacoes_monthly = transacoes.groupby('year_month').agg({'valor_transacao': 'sum'}).reset_index()
transacoes_monthly.rename(columns={'valor_transacao': 'volume_total'}, inplace=True)

# Define as datas mínima e máxima do conjunto de transações
min_date_str = transacoes['data_transacao'].min().strftime('%d/%m/%Y')
max_date_str = transacoes['data_transacao'].max().strftime('%d/%m/%Y')

# Faz a contagem de transeções por mês
transacoes_monthly_count = transacoes.groupby('year_month').agg(num_transacoes=('valor_transacao_abs', 'count')).reset_index()

# Combina os dados de volume e contagem em um único DataFrame e criar a coluna 'date'
df_transactions = pd.merge(transacoes_monthly, transacoes_monthly_count, on='year_month', how='left')
df_transactions['date'] = pd.to_datetime(df_transactions['year_month'])

# %% 5.2 Seção IPCA

# Vai resgatar os dados diretamente do site oficial
url_ipca = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json&dataInicial={min_date_str}&dataFinal={max_date_str}"
response = requests.get(url_ipca)
ipca = pd.DataFrame(response.json()) if response.status_code == 200 else None
if ipca is None:
    raise Exception("Erro ao acessar os dados do IPCA")
ipca['date'] = pd.to_datetime(ipca['data'], dayfirst=True)
ipca['ipca'] = ipca['valor'].astype(float)
ipca['year_month'] = ipca['date'].dt.to_period('M').astype(str)

# Mescla os dados do IPCA com os dados de transações
df_ipca = pd.merge(df_transactions, ipca[['year_month', 'ipca']], on='year_month', how='left')

# Impressões
print("IPCA e Volume Total:\n", df_ipca.head(), "\n")
print(f"Corr. IPCA x Volume Total: {df_ipca['volume_total'].corr(df_ipca['ipca']):.4f}")

# Série Temporal Normalizada
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_ipca[['volume_total_scaled', 'num_transacoes_scaled', 'ipca_scaled']] = scaler.fit_transform(
    df_ipca[['volume_total', 'num_transacoes', 'ipca']]
)

plt.figure(figsize=(12,6))
plt.plot(df_ipca['date'], df_ipca['volume_total_scaled'], marker='o', label='Volume Total')
plt.plot(df_ipca['date'], df_ipca['num_transacoes_scaled'], marker='o', label='Num. Transações')
plt.plot(df_ipca['date'], df_ipca['ipca_scaled'], marker='o', label='IPCA')
plt.title('Série Temporal Normalizada (IPCA)')
plt.xlabel('Data')
plt.ylabel('Valores Normalizados')
plt.legend()
plt.grid(True)
plt.figtext(0.5, 0.01, "Comentário: Série temporal normalizada comparando volume total, número de transações e IPCA.", ha="center", fontsize=10, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# Heatmap da Matriz de Correlação
plt.figure(figsize=(8,6))
corr_ipca = df_ipca[['volume_total', 'num_transacoes', 'ipca']].corr()
sns.heatmap(corr_ipca, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação (IPCA)')
plt.figtext(0.5, 0.01, "Comentário: Matriz de correlação entre volume total, número de transações e IPCA.", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.show()

# Pairplot para Visualização Conjunta
sns.pairplot(df_ipca[['volume_total', 'num_transacoes', 'ipca']])
plt.suptitle('Pairplot: Volume, Transações e IPCA', y=1.02)
plt.figtext(0.5, 0.01, "Comentário: Pairplot exibindo as distribuições e relações entre volume total, número de transações e IPCA.", ha="center", fontsize=10, color="gray")
plt.show()

# %% 5.3 Seção SELIC

# Vai resgatar os dados diretamente do site oficial
url_selic = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json&dataInicial={min_date_str}&dataFinal={max_date_str}"
response_selic = requests.get(url_selic)
selic = pd.DataFrame(response_selic.json()) if response_selic.status_code == 200 else None
if selic is None:
    raise Exception("Erro ao acessar os dados da SELIC")
selic['date'] = pd.to_datetime(selic['data'], dayfirst=True)
selic['selic'] = selic['valor'].astype(float)
selic['year_month'] = selic['date'].dt.to_period('M').astype(str)

# Mescla os dados da SELIC com os dados de transações
df_selic = pd.merge(df_transactions, selic[['year_month', 'selic']], on='year_month', how='left')

# Impressões
print("SELIC e Volume Total:\n", df_selic.head(), "\n")
print(f"Corr. SELIC x Volume Total: {df_selic['volume_total'].corr(df_selic['selic']):.4f}")

# Série Temporal Normalizada
scaler = MinMaxScaler()
df_selic[['volume_total_scaled', 'num_transacoes_scaled', 'selic_scaled']] = scaler.fit_transform(
    df_selic[['volume_total', 'num_transacoes', 'selic']]
)

plt.figure(figsize=(12,6))
plt.plot(df_selic['date'], df_selic['volume_total_scaled'], marker='o', label='Volume Total')
plt.plot(df_selic['date'], df_selic['num_transacoes_scaled'], marker='o', label='Num. Transações')
plt.plot(df_selic['date'], df_selic['selic_scaled'], marker='o', label='SELIC')
plt.title('Série Temporal Normalizada (SELIC)')
plt.xlabel('Data')
plt.ylabel('Valores Normalizados')
plt.legend()
plt.grid(True)
plt.figtext(0.5, 0.01, "Comentário: Série temporal normalizada comparando volume total, número de transações e SELIC.", ha="center", fontsize=10, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# Heatmap da Matriz de Correlação
plt.figure(figsize=(8,6))
corr_selic = df_selic[['volume_total', 'num_transacoes', 'selic']].corr()
sns.heatmap(corr_selic, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação (SELIC)')
plt.figtext(0.5, 0.01, "Comentário: Matriz de correlação entre volume total, número de transações e SELIC.", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.show()

# Pairplot para Visualização Conjunta
sns.pairplot(df_selic[['volume_total', 'num_transacoes', 'selic']])
plt.suptitle('Pairplot: Volume, Transações e SELIC', y=1.02)
plt.figtext(0.5, 0.01, "Comentário: Pairplot exibindo as distribuições e relações entre volume total, número de transações e SELIC.", ha="center", fontsize=10, color="gray")
plt.show()

# %% 5.4 Seção ICC

# Vai resgatar os dados diretamente do site oficial
url_icc = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.4393/dados?formato=json"
response_icc = requests.get(url_icc)
icc = pd.DataFrame(response_icc.json()) if response_icc.status_code == 200 else None
if icc is None:
    raise Exception("Erro ao acessar os dados do ICC")
icc['date'] = pd.to_datetime(icc['data'], dayfirst=True)
icc['icc'] = icc['valor'].astype(float)
icc['year_month'] = icc['date'].dt.to_period('M').astype(str)

# Mescla os dados do ICC com os dados de transações
df_icc = pd.merge(df_transactions, icc[['year_month', 'icc']], on='year_month', how='left')

# Impressões
print("ICC e Volume Total:\n", df_icc.head(), "\n")
print(f"Corr. ICC x Volume Total: {df_icc['volume_total'].corr(df_icc['icc']):.4f}")

# Série Temporal Normalizada
scaler = MinMaxScaler()
df_icc[['volume_total_scaled', 'num_transacoes_scaled', 'icc_scaled']] = scaler.fit_transform(
    df_icc[['volume_total', 'num_transacoes', 'icc']]
)

plt.figure(figsize=(12,6))
plt.plot(df_icc['date'], df_icc['volume_total_scaled'], marker='o', label='Volume Total')
plt.plot(df_icc['date'], df_icc['num_transacoes_scaled'], marker='o', label='Num. Transações')
plt.plot(df_icc['date'], df_icc['icc_scaled'], marker='o', label='ICC')
plt.title('Série Temporal Normalizada (ICC)')
plt.xlabel('Data')
plt.ylabel('Valores Normalizados')
plt.legend()
plt.grid(True)
plt.figtext(0.5, 0.01, "Comentário: Série temporal normalizada comparando volume total, número de transações e ICC.", ha="center", fontsize=10, color="gray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()

# Heatmap da Matriz de Correlação
plt.figure(figsize=(8,6))
corr_icc = df_icc[['volume_total', 'num_transacoes', 'icc']].corr()
sns.heatmap(corr_icc, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação (ICC)')
plt.figtext(0.5, 0.01, "Comentário: Matriz de correlação entre volume total, número de transações e ICC.", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.show()

# Pairplot para Visualização Conjunta
sns.pairplot(df_icc[['volume_total', 'num_transacoes', 'icc']])
plt.suptitle('Pairplot: Volume, Transações e ICC', y=1.02)
plt.figtext(0.5, 0.01, "Comentário: Pairplot exibindo as distribuições e relações entre volume total, número de transações e ICC.", ha="center", fontsize=10, color="gray")
plt.show()
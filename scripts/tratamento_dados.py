import pandas as pd
import re
from datetime import datetime

# %%  Leitura dos arquivos

agencias = pd.read_csv('agencias.csv')
clientes = pd.read_csv('clientes.csv')
colaborador_agencia = pd.read_csv('colaborador_agencia.csv')
colaboradores = pd.read_csv('colaboradores.csv')
contas = pd.read_csv('contas.csv')
propostas_credito = pd.read_csv('propostas_credito.csv')
transacoes = pd.read_csv('transacoes.csv')

# %% Funções

# Formata o CEP para o padrão XXXXX-XXX
def format_cep(cep):
    cep = str(cep).zfill(8)  # Garante que tenha 8 dígitos
    cep = re.sub(r'\D', '', cep)  # Remove caracteres não numéricos
    return f"{cep[:5]}-{cep[5:]}" if len(cep) == 8 else None

# Calcula a idade de forma precisa considerando ano, mês e dia
def calcular_idade(data_nascimento):
    if pd.notna(data_nascimento):
        nascimento = pd.to_datetime(data_nascimento, errors='coerce')
        if pd.notna(nascimento):
            hoje = datetime.now()
            idade = hoje.year - nascimento.year - ((hoje.month, hoje.day) < (nascimento.month, nascimento.day))
            return idade
    return None

# Extrai o ano e mês de uma data
def extrair_ano_mes(data):
    if pd.notna(data):
        return pd.to_datetime(data, errors='coerce').strftime('%Y-%m')
    return None

# Exibe o número de valores nulos em cada coluna de um DataFrame
def checar_valores_nulos(df, nome_df):
    nulos = df.isnull().sum()
    print(f"Valores nulos em {nome_df}:")
    print(nulos[nulos > 0])
    print("-" * 40)
    
# Extrai o CEP a partir de um campo de endereço
def extrair_cep_endereco(endereco):
    match = re.search(r'\d{5}-\d{3}', endereco)
    if match:
        return match.group(0)
    return None

# %% Processamento de agencias.csv

agencias['cep'] = agencias['endereco'].apply(extrair_cep_endereco) # Extrai o CEP do campo 'endereco' e cria a coluna 'cep'
agencias.drop(columns=['endereco'], inplace=True) # Remove a coluna 'endereco'
agencias.to_csv('agencias_processado.csv', index=False) # Salva o DataFrame processado em um novo arquivo CSV
agencias_processado = pd.read_csv('agencias_processado.csv') # Lê o arquivo processado
checar_valores_nulos(agencias_processado, 'agencias_processado') # Verifica se há valores nulos

# %% Processamento de clientes.csv

clientes['ano_mes_inclusao'] = clientes['data_inclusao'].apply(extrair_ano_mes) # Extrai ano e mês da data de inclusão e armazena em uma nova coluna
clientes['idade'] = clientes['data_nascimento'].apply(calcular_idade) # Calcula a idade com base na data de nascimento
clientes['cep'] = clientes['cep'].apply(format_cep) # Formata o CEP para o padrão 'XXXXX-XXX'
clientes.drop(columns=['data_inclusao', 'data_nascimento', 'endereco', 'cpfcnpj', 'email'], inplace=True) # Remove colunas desnecessárias após o processamento
clientes.rename(columns={'ano_mes_inclusao': 'data_inclusao'}, inplace=True) # Renomeia a coluna 'ano_mes_inclusao' para 'data_inclusao'
clientes.to_csv('clientes_processado.csv', index=False)
clientes_processado = pd.read_csv('clientes_processado.csv')
checar_valores_nulos(clientes_processado, 'clientes_processado')

# %% Processamento de colaboradores.csv

colaboradores = colaboradores.merge(colaborador_agencia[['cod_colaborador', 'cod_agencia']], on='cod_colaborador', how='left') # Realiza o merge para associar cada colaborador à sua agência
colaboradores['idade'] = colaboradores['data_nascimento'].apply(calcular_idade)
colaboradores['cep'] = colaboradores['cep'].apply(format_cep)
colaboradores.drop(columns=['data_nascimento', 'endereco', 'cpf', 'email'], inplace=True)
colaboradores.to_csv('colaboradores_processado.csv', index=False)
colaboradores_processado = pd.read_csv('colaboradores_processado.csv')
checar_valores_nulos(colaboradores_processado, 'colaboradores_processado')

# %% Processamento de contas.csv

contas['ano_mes_abertura'] = contas['data_abertura'].apply(extrair_ano_mes)
contas['ano_mes_ultimo_lancamento'] = contas['data_ultimo_lancamento'].apply(extrair_ano_mes)
contas.drop(columns=['data_abertura', 'data_ultimo_lancamento'], inplace=True)
contas.rename(columns={'ano_mes_abertura': 'data_abertura', 'ano_mes_ultimo_lancamento': 'data_ultimo_lancamento'}, inplace=True)
contas.to_csv('contas_processado.csv', index=False)
contas_processado = pd.read_csv('contas_processado.csv')
checar_valores_nulos(contas_processado, 'contas_processado')

# %% Processamento de propostas_credito.csv

propostas_credito['ano_mes_entrada_proposta'] = propostas_credito['data_entrada_proposta'].apply(extrair_ano_mes)
propostas_credito.drop(columns=['data_entrada_proposta'], inplace=True)
propostas_credito.rename(columns={'ano_mes_entrada_proposta': 'data_entrada_proposta'}, inplace=True)
propostas_credito.to_csv('propostas_credito_processado.csv', index=False)
propostas_credito_processado = pd.read_csv('propostas_credito_processado.csv')
checar_valores_nulos(propostas_credito_processado, 'propostas_credito_processado')

# %% Processamento de transacoes.csv

transacoes['ano_mes_transacao'] = transacoes['data_transacao'].apply(extrair_ano_mes)
transacoes['valor_transacao_abs'] = transacoes['valor_transacao'].abs() # Cria uma coluna com o valor absoluto da transação

# %% Classificação de nome_transacao

# Dicionário que agrupa as transações em 'Entrada' e 'Saída'
transacao_grupo = {
    'Entrada': ['Pix - Recebido', 'TED - Recebido', 'DOC - Recebido', 'Depósito em espécie', 'Estorno de Debito', 'Transferência entre CC - Crédito'],
    'Saída': ['Saque', 'Pix Saque', 'Compra Débito', 'Compra Crédito', 'DOC - Realizado', 'Pix - Realizado', 'TED - Realizado', 'Pagamento de boleto', 'Transferência entre CC - Débito']
}

# Dicionário para simplificar o nome das transações.
transacao_simplificada = {
    'Pix' : ['Pix - Recebido','Pix - Realizado'],
    'TED' : ['TED - Recebido', 'TED - Realizado'],
    'DOC' : ['DOC - Recebido', 'DOC - Realizado'],
    'Transferência entre CC' : ['Transferência entre CC - Crédito', 'Transferência entre CC - Débito'],
    'Depósito em espécie' : 'Depósito em espécie',
    'Estorno de Debito' : 'Estorno de Debito',
    'Saque': ['Saque'],
    'Pix Saque': ['Pix Saque'],
    'Compra Débito': ['Compra Débito'],
    'Compra Crédito': ['Compra Crédito'],
    'Pagamento de boleto': ['Pagamento de boleto']
    }

# Classifica a transação como 'Entrada', 'Saída' ou 'Outro'
def categorizar_transacao(nome):
    for categoria, lista in transacao_grupo.items():
        if nome in lista:
            return categoria
    return 'Outro'

#  Simplifica o nome da transação de acordo com o dicionário
def simplificar_transacao(nome):
    for categoria, lista in transacao_simplificada.items():
        if nome in lista:
            return categoria
    return 'Outro'

# Aplica a simplificação e a categorização nos nomes das transações originais
transacoes['transacao_simplificada'] = transacoes['nome_transacao'].apply(simplificar_transacao)
transacoes['categoria_transacao'] = transacoes['nome_transacao'].apply(categorizar_transacao)

transacoes.drop(columns=['data_transacao', 'nome_transacao'], inplace=True)
transacoes.rename(columns={'ano_mes_transacao': 'data_transacao', 'transacao_simplificada':'nome_transacao'}, inplace=True)
transacoes.to_csv('transacoes_processado.csv', index=False)
transacoes_processado = pd.read_csv('transacoes_processado.csv')
checar_valores_nulos(transacoes_processado, 'transacoes_processado')

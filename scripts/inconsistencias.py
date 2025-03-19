import pandas as pd

# %% Leitura dos arquivos processados
agencias = pd.read_csv('agencias_processado.csv')
clientes = pd.read_csv('clientes_processado.csv')
colaboradores = pd.read_csv('colaboradores_processado.csv')
contas = pd.read_csv('contas_processado.csv')
propostas_credito = pd.read_csv('propostas_credito_processado.csv')
transacoes = pd.read_csv('transacoes_processado.csv')
colaborador_agencia = pd.read_csv('colaborador_agencia.csv')

# %% Processamento da tabela "contas"
# Regra: os registros devem ter cod_cliente, cod_agencia e cod_colaborador válidos
mask_contas = (
    contas['cod_cliente'].isin(clientes['cod_cliente']) &
    contas['cod_agencia'].isin(agencias['cod_agencia']) &
    contas['cod_colaborador'].isin(colaboradores['cod_colaborador'])
)

contas_clean = contas[mask_contas].copy()
contas_inconsistentes = contas[~mask_contas].copy()

# %% Processamento da tabela "propostas_credito"
# Regra: os registros devem ter cod_cliente e cod_colaborador válidos
mask_propostas = (
    propostas_credito['cod_cliente'].isin(clientes['cod_cliente']) &
    propostas_credito['cod_colaborador'].isin(colaboradores['cod_colaborador'])
)

propostas_credito_clean = propostas_credito[mask_propostas].copy()
propostas_credito_inconsistentes = propostas_credito[~mask_propostas].copy()

# %% Processamento da tabela "colaborador_agencia"
# Regra: os registros devem ter cod_agencia e cod_colaborador válidos
mask_colab_agencia = (
    colaborador_agencia['cod_agencia'].isin(agencias['cod_agencia']) &
    colaborador_agencia['cod_colaborador'].isin(colaboradores['cod_colaborador'])
)

colaborador_agencia_clean = colaborador_agencia[mask_colab_agencia].copy()
colaborador_agencia_inconsistentes = colaborador_agencia[~mask_colab_agencia].copy()

# %% Processamento da tabela "transacoes"
# Regra: os registros devem ter num_conta válido, isto é, existente na versão limpa de "contas"
mask_transacoes = transacoes['num_conta'].isin(contas_clean['num_conta'])
transacoes_clean = transacoes[mask_transacoes].copy()
transacoes_inconsistentes = transacoes[~mask_transacoes].copy()

# %% Salvando os novos arquivos CSV sem inconsistências e os registros inconsistentes

# Tabela "contas"
contas_clean.to_csv('contas_sem_inconsistencias.csv', index=False)
contas_inconsistentes.to_csv('contas_inconsistentes.csv', index=False)

# Tabela "propostas_credito"
propostas_credito_clean.to_csv('propostas_credito_sem_inconsistencias.csv', index=False)
propostas_credito_inconsistentes.to_csv('propostas_credito_inconsistentes.csv', index=False)

# Tabela "colaborador_agencia"
colaborador_agencia_clean.to_csv('colaborador_agencia_sem_inconsistencias.csv', index=False)
colaborador_agencia_inconsistentes.to_csv('colaborador_agencia_inconsistentes.csv', index=False)

# Tabela "transacoes"
transacoes_clean.to_csv('transacoes_sem_inconsistencias.csv', index=False)
transacoes_inconsistentes.to_csv('transacoes_inconsistentes.csv', index=False)

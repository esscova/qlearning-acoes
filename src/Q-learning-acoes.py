
import yfinance as yf
import random
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime


# Obtenção de dados
ticker = 'PETR4.SA'
petro = yf.download(ticker, start='2018-01-01', end='2023-12-31')
dados = pd.DataFrame(petro)
dados.head(3)

# Visualização dos dados

fig = go.Figure(
    data=[go.Candlestick(x=dados.index,
                         open=dados['Open'],
                         high=dados['High'],
                         low=dados['Low'],
                         close=dados['Close']
                         )])
fig.show()


# ## Configurações

# **Hiperparâmetros**


episodios = 1000
alfa = 0.1  # Taxa de aprendizado
gama = 0.99  # Taxa de recompensa
epsilon = 0.1  # Exploração vs Exploração


# **Ambiente de negociação**
precos = dados.Close.values 
acoes = ['comprar', 'vender','manter'] # Ações de negociação
saldo_inicial = 1000
num_acoes_inicial = 0

# executar os passos do robo trading
def executar_acao (estado, acao, saldo, num_acoes, preco):

    # comprar
    if acao == 0:
        if saldo >= preco:
            num_acoes += 1
            saldo -= preco

    # vender
    elif acao == 1:
        if num_acoes > 0:
            num_acoes -= 1
            saldo += preco

    # lucro
    lucro = saldo + num_acoes*preco - saldo_inicial

    return (saldo, num_acoes, lucro) 


# ## Algoritmo Q-learning
q_tabela = np.zeros((len(precos), len(acoes)))

for _ in range(episodios):

  saldo = saldo_inicial
  num_acoes = num_acoes_inicial

  for i, preco in enumerate(precos[:-1]):
    estado = i

    if np.random.random() < epsilon:
      acao = random.choice(range(len(acoes)))
    else:
      acao = np.argmax(q_tabela[estado])

    saldo, num_acoes, lucro = executar_acao(estado, acao, saldo, num_acoes,preco)
    prox_estado = i + 1

    q_tabela[estado][acao] += alfa * (lucro + gama * np.max(q_tabela[prox_estado]) - q_tabela[estado][acao])
print('Treinamento concluído')


# Execução do algoritmo treinado
saldo = saldo_inicial
num_acoes = num_acoes_inicial

for i, preco in enumerate(precos[:-1]):
  estado = iacao = np.argmax(q_tabela[estado])
  saldo, num_acoes, _ = executar_acao(estado, acao, saldo, num_acoes, preco)

print('Execução concluída')


# Resultados

print(f'\nO modelo treinado está acumulando um total de: {num_acoes} ações com ticker {ticker}')
print(f'\nÚltimo preço de fechamento: R${round(precos[-1], 2)}')


# **Vendendo todas as ações no último preço de fechamento**
saldo += num_acoes * precos[-1]
lucro = saldo - saldo_inicial
lucro_final = round(lucro, 2)


# **Relatório**
print(f'\nRelatório de Negociação:')
print(f'\nSaldo inicial: {saldo_inicial}')
print(f'Saldo final: {round(saldo,2)}')
print(f'Lucro: {lucro_final}')


# **Exportação de dados para csv**
dados.to_csv('data/PETR4.csv', index=False)


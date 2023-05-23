# %%
# Importando as bibliotecas do projeto
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import probplot
import matplotlib.pyplot as plt
# %%
# importando o dataset
df = pd.read_csv('data/dados_projeto.csv', sep=';')
# %%
df.shape
# %%
df.head()
# %%
# Obtendo e avaliando as estatisticas descritivas
df.describe()
# %%
# Construindo um box-plot
ax = sns.boxplot(data=df, x='Y', orient='h', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Box plot', fontsize=20)
ax.set_xlabel('Consumo de cerveja (litros)', fontsize=16)
ax
# %%
# Identifique se existe uma relação linear entre as variáveis Y e X
ax = sns.lmplot(x='X', y='Y', data=df)
ax.fig.set_size_inches(12,6)
ax.fig.suptitle('Reta de Regressão = Gasto X Renda', fontsize=16, y=1.02)
ax.set_xlabels('Renda das Famílias', fontsize=14)
ax.set_ylabels('Gasto das Famílias', fontsize=14)
ax
# %%
df.corr()
# %%
# Preparando os dados para estimar um modelo de regressão linear simples
Y = df.Y
X = sm.add_constant(df.X)
# %%
resultado_regressao = sm.OLS(Y, X).fit()
# %%
resultado_regressao.summary()
# %%
# Obter o Y Previsto
df['Y_previsto'] = resultado_regressao.predict()
df.head()
# %%
# Qual seria o consumo de cerveja para um dia com temperatura média de 42C?
resultado_regressao.predict([1,42])[0]
# %%
# Obtendo os resíduos da regressão
df['Residuos'] = resultado_regressao.resid
df.head()
# %%
# Plotando um gráfico com os resíduos da regressão com o Y Previsto
ax = sns.scatterplot(x=df.Y_previsto, y=df.Residuos)
ax.figure.set_size_inches(12,6)
ax.set_title('Resíduos vs Y_Previsto', fontsize=18)
ax.set_xlabel('Y_Previsto', fontsize=14)
ax.set_ylabel('Resíduos', fontsize=14)
ax
# %%
# Obtenha o QQPlot dos resíduos
(_,(_,_,_)) = probplot(df.Residuos, plot=plt)
# %%
resultado_regressao.rsquared
# %%

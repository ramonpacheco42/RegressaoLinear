# %%
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
# %%
dados = pd.read_csv('data/dados.csv')
# %%
dados
# %%
# Gerando uma amostra aleatória para facilitar o entendimento
amostra = dados.query('Renda < 5000').sample(n = 20, random_state = 101)
# %%
# Obtendo a matriz de covariância
amostra[['Idade','Renda', 'Anos de Estudo', 'Altura']].cov()
# %%
# Identificando as variâncias na diagonal principal da matriz
amostra.Idade.var()
# %%
# Varificando a existência de uma associação linear negativa
x = amostra.Renda
y = amostra.Idade

ax = sns.scatterplot(x=x, y=y)
ax.figure.set_size_inches(10,6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors = 'black', linestyles = 'dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors = 'black', linestyles = 'dashed')
# %%
x = amostra.Renda
y = amostra['Anos de Estudo']

ax = sns.scatterplot(x=x, y=y)
ax.figure.set_size_inches(10,6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors = 'black', linestyles = 'dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors = 'black', linestyles = 'dashed')
# %%
x = amostra.Idade
y = amostra.Altura

ax = sns.scatterplot(x=x, y=y)
ax.figure.set_size_inches(10,6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors = 'black', linestyles = 'dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors = 'black', linestyles = 'dashed')
# %%
# Coeficiente de correlação de Pearson
# Obtendo Sxy
s_xy = dados[['Altura', 'Renda']].cov()
s_xy
# %%
s_xy = s_xy.Altura.loc['Renda']
s_xy
# %%
s_x = dados.Altura.std()
s_y = dados.Renda.std()
# %%
# Obtendo o coeficiente de correlação
r_xy = s_xy / (s_x * s_y)
r_xy
# %%
# Obtendo uma matriz de correlação com o Pandas
dados[['Altura', 'Renda']].corr()
# %%
x = amostra.Renda
y = amostra.Altura

ax = sns.scatterplot(x=x, y=y)
ax.figure.set_size_inches(10,6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors = 'black', linestyles = 'dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors = 'black', linestyles = 'dashed')
# %%
dataset = {
    'Y': [670, 220, 1202, 188, 1869, 248, 477, 1294, 816, 2671, 1403, 1586, 3468, 973, 701, 5310, 10950, 2008, 9574, 28863, 6466, 4274, 6432, 1326, 1423, 3211, 2140], 
    'X': [1.59, 0.56, 2.68, 0.47, 5.2, 0.58, 1.32, 3.88, 2.11, 5.53, 2.6, 2.94, 6.62, 1.91, 1.48, 10.64, 22.39, 4.2, 21.9, 59.66, 14.22, 9.57, 14.67, 3.28, 3.49, 6.94, 6.25]
}
# %%
dataset = pd.DataFrame(dataset)
Y = dataset.Y
X = sm.add_constant(dataset.X)
# %%
regressao_linear = sm.OLS(Y,X, missing='drop').fit()
# %%
beta_1, beta_2 = regressao_linear.params
print(beta_1)
print(beta_2)
# %%
dataset['Y_previsto'] = beta_1 + beta_2 * dataset.X
dataset.head(10)
# %%
dataset['Y_previsto_statsmodels'] = regressao_linear.predict()
dataset
# %%
dataset.drop(['Y_previsto_statsmodels'], axis = 1, inplace=True)
# %%
dataset
# %%
# Estimando o gasto das familias fora da amostra
def prever(x):
    return beta_1 + beta_2 * x

prever(2.11)
# %%
# Estimando o gasto das familias fora da amostra statsmodel
regressao_linear.predict([1,2.14])[0]
# %%
dataset['u'] = dataset.Y - dataset.Y_previsto
dataset
# %%
dataset['Residuos'] = regressao_linear.resid
dataset
# %%
dataset.drop(['u'], axis = 1, inplace = True)
# %%
ax = sns.scatterplot(x=dataset.X, y=dataset.Residuos)
ax.figure.set_size_inches(12,6)
ax.set_title('Resíduos vs Variável Idependente', fontsize=18)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Resíduos', fontsize=14)
ax
# %%
ax = sns.scatterplot(x=dataset.Y_previsto, y=dataset.Residuos)
ax.figure.set_size_inches(12,6)
ax.set_title('Resíduos vs Variável Idependente', fontsize=18)
ax.set_xlabel('Y_Previsto', fontsize=14)
ax.set_ylabel('Resíduos', fontsize=14)
ax
# %%
ax = sns.scatterplot(x=dataset.Y_previsto, y=dataset.Residuos**2)
ax.figure.set_size_inches(12,6)
ax.set_title('Resíduos vs Variável Idependente', fontsize=18)
ax.set_xlabel('Y_Previsto', fontsize=14)
ax.set_ylabel('Resíduosˆ2', fontsize=14)
ax
# %%

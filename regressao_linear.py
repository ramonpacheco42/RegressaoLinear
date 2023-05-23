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
# Coeficiente de determinação R2
sqe = dataset.Residuos.apply(lambda u: u**2).sum()
sqe
# %%
# Obtendo a soma dos quadrados residuals pelo statsmodel
regressao_linear.ssr
# %%
# Soma dos quadrados total SQT
sqt = dataset.Y.apply(lambda y: (y - dataset.Y.mean())**2).sum()
sqt
# %%
# Soma dos quadrados da regressão SQR
sqr = dataset.Y_previsto.apply(lambda y: (y - dataset.Y.mean())**2).sum()
sqr
# %%
# Obtendo SQR pelo statsmodel
regressao_linear.ess
# %%
# Obtendo o R2
sqr/sqt
# %%
# Obtendo o R2 pelo statsmodel
regressao_linear.rsquared
# %%
# Obtendo o R2 pela correlação
0.998395**2
# %%
# Testes aplicados a modelos de regressão
print(regressao_linear.summary())
# %%
# Calculando o erro quadrádico médio
n = len(df)
eqm = sqe / (n-2)
eqm
  # %%
# Calculando EQM pelo Statsmodel
eqm = regressao_linear.mse_resid
eqm
# %%
# Teste de hipótese para nulidade do coeficiente angular
# Calculando s
s = np.sqrt(regressao_linear.mse_resid)
s
# %%
# Calculando a ∑(x1 - x2)2
soma_desvio = dataset.X.apply(lambda x: (x - dataset.X.mean())**2).sum()
soma_desvio
# %%
s_beta_2 = s / np.sqrt(soma_desvio)
s_beta_2
# %%
from scipy.stats import t as t_student
# %%
confianca = 0.95
significancia = 1 - confianca
# %%
graus_de_liberdade = regressao_linear.df_resid
graus_de_liberdade
# %%
probabilidade = (0.5 + (confianca / 2))
probabilidade
# %%
t_alpha_2 = t_student.ppf(probabilidade,graus_de_liberdade)
t_alpha_2
# %%
t = (beta_2 - 0) / s_beta_2
t 
# %%
regressao_linear.tvalues[1]
# %%
t <= t_alpha_2
# %%
t >= t_alpha_2
# %%
# Rejeitar H0 se o valor P< a
p_valor = 2 * (t_student.sf(t, graus_de_liberdade))
p_valor
# %%
p_valor = regressao_linear.pvalues[1] 
p_valor
# %%
p_valor <= significancia
# %%
print(regressao_linear.summary())   
# %%
# Teste F -> Teste para multiparametros
regressao_linear.mse_model
# %%
regressao_linear.mse_resid
# %%
regressao_linear.mse_model / regressao_linear.mse_resid
# %%
# Outra forma de obter estatistica F
regressao_linear.fvalue
# %%
# Obtendo o p-valor
regressao_linear.f_pvalue
# %%
# Outros Testes
# Omnibus
from scipy.stats import normaltest
statistic, p_Valor = normaltest(dataset.Residuos)
# %%
# Como o valor é menor de 0.05 rejeitamos a hipotese nula, o que quer dizer que o dataframe
# Não se conporta como uma distribuição normal.
p_valor <= 0.05
# %%
from scipy.stats import probplot
import matplotlib.pyplot as plt
(_,(_,_,_)) = probplot(dataset.Residuos, plot=plt)
# %%
from sklearn.preprocessing import StandardScaler
# Crie uma instância do StandardScaler
scaler = StandardScaler()

# Ajuste e transforme os dados
dados_normalizados = scaler.fit_transform(dataset)
# %%
import scipy.stats as stats
import matplotlib.pyplot as plt

# Transforme a matriz bidimensional em um array unidimensional
dados_unidimensionais = dados_normalizados.flatten()

(_,(_,_,_)) = probplot(dados_unidimensionais, plot=plt)
# %%
import matplotlib.pyplot as plt
# Plote o histograma dos dados unidimensionais
plt.hist(dados_unidimensionais, bins=10)
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.title('Histograma dos dados unidimensionais')
plt.show()
# %%
dados.Altura.hist(bins=50)
# %%
(_,(_,_,_)) = probplot(dados.Altura, plot=plt)
# %%
# Verificando a simetria
from scipy.stats import skew
S = skew(dataset.Residuos)
S
# %%
# Verificando a Curtose
from scipy.stats import kurtosis
C = 3 + kurtosis(dataset.Residuos)
C
# %%
# Teste de Jarque-Bera (Statsmodels)
n = len(dataset)
JB = (n/6.) * (S ** 2 + (1/4.)* (C-3)**2)
JB
# %%
# Teste de Jarque-Bera Verificando se é uma distribuição normal. 
from scipy.stats import chi2
p_valor = chi2.sf(JB,2)
p_valor
# %%
p_valor <= 0.05
# %%
n = len(dataset)
JB = (n - 1 /6.) * (S ** 2 + (1/4.)* (C-3)**2)
JB
# %%
p_valor = chi2.sf(JB,2)
p_valor
# %%
p_valor <= 0.05
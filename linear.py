# Podemos considerar esses dados abaixo
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

dados = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [1, 3, 1.8, 3.5, 4, 4.6]
})

dados.head(2)

# Visualizando esses pontos graficamente, podemos traçar uma reta que passa por esses pontos

fig, ax = plt.subplots()

ax.scatter(dados.X, dados.Y)
ax.plot(dados.X, dados.y_reta, '--r')
ax.scatter(dados.X, dados.y_reta)

plt.show()

# Nessa reta vermelha fizemos que y = x, então podemos escrever o y_reta como
dados['y_reta'] = dados.x
# Importando a regressão linear

# Criando o regressor
reg = LinearRegression().fit(dados.X.values.reshape(-1, 1), dados.Y)

# Visualizando o coeficiente angular
reg.coef_

# e o coeficiente linear
reg.intercept_

# Importando o dataset
x, y = load_iris(return_X_y=True, as_frame=True)

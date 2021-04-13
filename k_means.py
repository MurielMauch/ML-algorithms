import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

df_dataset_cluster = pd.read_csv('./data/cluster.dat', sep='\s+', index_col=None, header=None)
df_dataset_cluster.columns = ['x', 'y']


def normalizar_dados(dataset):
    # define o scaler a ser usado
    scaler = StandardScaler()

    # aplica o normalizador
    dataset = scaler.fit_transform(dataset)
    dataframe = pd.DataFrame(dataset)

    return dataframe


def inicializar_centroides(dataset, K):
    indices = np.random.permutation(range(dataset.shape[0]))

    indices = indices[0:K]
    centroides = dataset[indices, :]

    return centroides

def centroides_proximos(dataset, centroides):
      n = dataset.shape[0]
      K = centroides.shape[0]

      indice = np.zeros(n, dtype=int)
      somatorio_distancias = 0

      for i in range(n):
        distancia = np.sqrt(((dataset[i] - centroides) ** 2).sum(axis=1))
        somatorio_distancias += distancia
        indice[i] = np.argmin(distancia)

      media_distancias = somatorio_distancias / n

      return indice, media_distancias

def calcular_centroides(dataset, K, indices):
      n = dataset.shape[1]
      centroides = np.zeros([K, n])

      for i in range(K):
        grupo_centroides = dataset[indices==i]
        centroides[i, :] = grupo_centroides.mean(0)

      return centroides


def kmeans(dataset, K, centroides, max_iteracoes):
    m, n = dataset.shape

    indices = np.zeros(m)
    media_das_distancias = 0

    centro = centroides

    i = 0
    diff = 1

    while i < max_iteracoes and diff != 0:
        indices, media_distancias = centroides_proximos(dataset, centro)

        centro = calcular_centroides(dataset, K, indices);

        diff = (centroides - centro).sum() ** 2

        centroides = centro

        i += 1

    indices = centroides_proximos(dataset, centro)

    return centro, indices, media_distancias


def visualizar_dados(dataset, centroides, indices):
    cores = 5 * ['b', 'g', 'r', 'c', 'm', 'y']

    for i in range(centroides.shape[0]):
        plt.scatter(dataset[indices == i, 0], dataset[indices == i, 1], marker='.', label='Dados', color=cores[i], s=50)

        plt.scatter(centroides[i, 0], centroides[i, 1], marker='x', color='black', s=150, lw=3)

    plt.show()

valores_k = np.array([1,2,3,4])


ctr, id, media_das_distancias = 0, 0, 0

for i, k in enumerate(valores_k):
  print("\nK = ", k)

  centroides = inicializar_centroides(df_dataset_cluster.values, k)

  ctr, id, media_das_distancias = kmeans(df_dataset_cluster.values, k, centroides, 10)


visualizar_dados(df_dataset_cluster.values, ctr, id)

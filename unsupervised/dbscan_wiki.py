# Unlike the most well known K-mean, DBSCAN does not need to specify the number of clusters.
# It can automatically detect the number of clusters based on your input data and parameters

# arbitrary select a point p
# retrieve all points density-reachable from p based on EPS and minpts
# if p is a core point, a cluster is formed
# continue the process until all points have been processed

# implementação baseada no pseudo código disponível em
# DBSCAN. Wikipedia, 2021. Disponível em: <https://en.wikipedia.org/wiki/DBSCAN#Algorithm>.
# Acesso em: 10 de abril de 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style

style.use('ggplot')


# função para achar os pontos dentro de um raio
def encontre_vizinhos(dataset_rotulado, ponto: dict, eps):
    vizinhos = []
    for datapoint in dataset_rotulado.items():
        # distancia euclidiana usando norma euclidiana (norma L2)
        dist = np.linalg.norm(ponto.get('ponto') - datapoint[1].get('ponto'))
        if 0 < dist <= eps:
            vizinhos.append(datapoint)
    return vizinhos


def dbscan(dataset, eps, minpts):
    c = 0  # contador dos clusters
    vizinhanca = []
    dataset_rotulado = {}

    for index in range(len(dataset)):  # rotulando todos os pontos como indefinidos, ou seja, iguais a 0
        dataset_rotulado[index] = {
            'label': 0,
            'ponto': dataset[index]
        }

    for index in range(len(dataset)): # criando um dicionario para rotular cada ponto
        ponto = dataset_rotulado[index]
        if ponto.get('label') != 0:
            continue  # enquanto não visitarmos todos os pontos queremos continuar

        vizinhos = encontre_vizinhos(dataset_rotulado, ponto, eps)  # lista de pontos rotulados no formato dict

        if len(vizinhos) < minpts:  # rotulamos o ponto p como noise, atribuindo o valor -1
            dataset_rotulado[index]['label'] = -1
            continue

        c += 1  # rotulamos o próximo cluster

        dataset_rotulado[index]['label'] = c  # rotulamos o ponto com o cluster

        vizinhanca = vizinhos  # atribuindo à vizinhanca a nossa lista de vizinhos rotulados

        for internal_index in range(len(vizinhanca)):
            if vizinhanca[internal_index][1].get('label') == -1:  # se for noise, entra no cluster
                vizinhanca[internal_index][1]['label'] = c
            elif vizinhanca[internal_index][1].get('label') != 0:
                continue
            vizinhanca[internal_index][1]['label'] = c  # ???

            vizinhos_do_vizinho = encontre_vizinhos(dataset_rotulado, vizinhanca[internal_index][1], eps)
            if len(vizinhos_do_vizinho) >= minpts:
                vizinhanca += vizinhos_do_vizinho

    return dataset_rotulado


# definir labels para os clusters de pontos
eps = 150  # largura de banda maxima da vizinhanca
minpts = 34  # numero minimo de pontos para ser considerada uma vizinhanca

# lendo os dados
df_dataset_cluster = pd.read_csv('data/cluster.dat', sep='\s+', index_col=None, header=None)
df_dataset_cluster.columns = ['x', 'y']

# descrevendo o dataset
print(df_dataset_cluster.describe())
print(len(df_dataset_cluster))

dataset = df_dataset_cluster.to_numpy()

print('Classificando...')
classificacao = dbscan(dataset, eps, minpts)
print('Classificado!')

cores = 30*['black', 'green', 'brown', 'blue', 'purple', 'orange', 'yellow', 'red', 'magenta', 'pink', 'gold', 'gray',
            'silver', 'violet']

clusters = {}

print('Separando os clusters')
for i in range(len(classificacao)):
    datapoint = classificacao.get(i)
    cluster = datapoint.get('label')
    try:
        clusters[cluster].append(datapoint.get('ponto'))
    except:
        clusters[cluster] = []
        clusters[cluster].append(datapoint.get('ponto'))
print('Done!')

print(clusters)
print('Colorindo...')
for i in range(len(clusters)):
    for pontos in clusters.items():
        rows = []
        lines = []
        cor = cores[pontos[0]]
        for datapoint in pontos[1]:
            rows.append(datapoint[0])
            lines.append(datapoint[1])
        plt.scatter(rows, lines, c=cor, marker="*", s=5)
print('Bora plottar?')

plt.show()



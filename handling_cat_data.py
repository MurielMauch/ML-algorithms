import pandas as pd

# os dados que compõem o dataset são:
# @attribute Gender {Female,Male}                                                   Nominal
# @attribute Age numeric
# @attribute Height numeric
# @attribute Weight numeric
# @attribute family_history_with_overweight {yes,no}                                Nominal
# @attribute FAVC {yes,no}                                                          Nominal
# @attribute FCVC numeric
# @attribute NCP numeric
# @attribute CAEC {no,Sometimes,Frequently,Always}                                  Ordinal
# @attribute SMOKE {yes,no}                                                         Nominal
# @attribute CH2O numeric
# @attribute SCC {yes,no}                                                           Nominal
# @attribute FAF numeric
# @attribute TUE numeric
# @attribute CALC {no,Sometimes,Frequently,Always}                                  Ordinal
# @attribute MTRANS {Automobile,Motorbike,Bike,Public_Transportation,Walking}       Nominal
# @attribute NObeyesdad {Insufficient_Weight,Normal_Weight,                         Classe, vamos retirar...
#                        Overweight_Level_I,Overweight_Level_II,Obesity_Type_I,Obesity_Type_II,Obesity_Type_III}

df_dataset_obesidade = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv', sep=',', index_col=None)

# copiando o dataset para manter o formato do original
dataset_cp = df_dataset_obesidade.copy()

# retirando a coluna da classe
dataset_cp = dataset_cp.drop('NObeyesdad', axis=1)

# criando o dicionário para substituir os valores ordinais e nominais
ordinal_values = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
gender_values = {'Male': 0, 'Female': 1}
binary_values = {'no': 0, 'yes': 1}

dataset_cp['CAEC'] = dataset_cp.CAEC.map(ordinal_values)
dataset_cp['CALC'] = dataset_cp.CALC.map(ordinal_values)
dataset_cp['Gender'] = dataset_cp.Gender.map(gender_values)
dataset_cp['family_history_with_overweight'] = dataset_cp.family_history_with_overweight.map(binary_values)
dataset_cp['FAVC'] = dataset_cp.FAVC.map(binary_values)
dataset_cp['SMOKE'] = dataset_cp.SMOKE.map(binary_values)
dataset_cp['SCC'] = dataset_cp.SCC.map(binary_values)

# substituindo o valor nominal meio de transport e criando novas colunas com valores binários e sufixo MTRANS
dataset_cp = pd.get_dummies(dataset_cp, columns=['MTRANS'], prefix=['MTRANS'])

pre_processed_dataset = dataset_cp.copy()

# pre_processed_dataset.to_csv('data/pre_processed_dataset.csv', index=False)

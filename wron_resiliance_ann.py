import numpy as np
import pandas as pd

import re
import time

import io
from contextlib import redirect_stdout

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


"""

Função preprocessing: Limpa o conjunto de dados e separa as métricas da camada de entrada.
- Limpa os dados
- Separa as métricas
- Gera arquivo de nome "dataset.csv"

Entrada: Lista com o path dos arquivos

"""
def preprocessing(filenames):
  
  # cleaning the data and joining it in the 'dataset.txt' file
  data = ""
  for file in filenames:
    arq = open(file, 'r')
    data += arq.read()
    arq.close()
  
  regex = r"(;\n|;$)"
  clean_data = re.sub(regex, '\n', data)
  
  arq = open('data.txt', 'w')
  arq.write(clean_data)
  arq.close()
  
  # separating and normalizing the metrics
  dataframe = pd.read_csv('data.txt', header=None, sep=';')
  database = dataframe.values
  
  # taking the metric "Algebraic Connectivity"
  algebraic_connectivity = database[:, 4:5]
  
  # taking the metric "Natural Connectivity"
  natural_connectivity = database[:, 5:6]
  
  # taking the metric "DFT of Laplacian Entropy"
  dft_laplacian_entropy = database[:, 14:15]
  
  # taking the metric "Number of Nodes"
  list = [[18]] * len(algebraic_connectivity)
  num_nodes = np.array(list)
  
  # taking the metric "Number of Links"
  list = []
    
  for mtx_adj in database[:, 15:16]:
    count = 0
    for vet_adj in mtx_adj:
      for cel in vet_adj:
        if cel == '1':
          count += 1
    
    list.append([count])
  
  num_links = np.array(list)
  
  # taking the metric "Hub Degree"
  mtx_adj = []
  list = []
  i_complete = 0
  j_complete = 0
  count = 0
  links_count= 0
  max_connections = 0
  
  # building the superior matrix
  for np_adj in database[:, 15:16]:
    for str_adj in np_adj:
      vet_adj = str_adj.split(' ')
      count = 0
      links_count = 0
      max_connections = 0
      mtx_adj = []
      for i in range(18):
        i_complete = i
        j_complete = 17 - i
        lista = []
        for j in range(18):
          if i == j:
            lista.append(0)
            i_complete -= 1
          else:
            if i_complete >= 0:
              lista.append(-1)
              i_complete -= 1
            elif j_complete >= 0:
              lista.append(int(vet_adj[count]))
              count += 1
              j_complete -= 1
              
        mtx_adj.append(lista)
        
      # building the full matrix
      for i in range(18):
        for j in range(18):
          if mtx_adj[i][j] == -1:
            mtx_adj[i][j] = mtx_adj[j][i]
      
      for i in range(18):
        for j in range(18):
          if mtx_adj[i][j] == 1:
            links_count += 1           
        if links_count > max_connections:
          max_connections = links_count
        links_count = 0
      
      list.append([max_connections])
  
  hub_degree = np.array(list)
  
  # taking the class "Robustness for simple failures"
  simple_failures_robustness = database[:, 30:31]
  
  # taking the class "Robustness for double failures"
  double_failures_robustness = database[:, 31:32]
  
  dataset = []
  list = []  
  
  for i in range(len(database)):
    list.append(algebraic_connectivity[i].tolist()[0])
    list.append(natural_connectivity[i].tolist()[0])
    list.append(dft_laplacian_entropy[i].tolist()[0])
    list.append(num_nodes[i].tolist()[0])
    list.append(num_links[i].tolist()[0])
    list.append(hub_degree[i].tolist()[0])
    list.append(simple_failures_robustness[i].tolist()[0]) 
    list.append(double_failures_robustness[i].tolist()[0])
    dataset.append(list)
    list = []
    
  my_df = pd.DataFrame(dataset)
  my_df.to_csv('dataset.csv', index=False, header=False)

"""

Função which_class: Diz qual à classe um exemplo pertence.
Entrada: Exemplo do dataset e coluna com a classe
Saída: Classe

"""
 def whichclass(exemple, col):
  
  if exemple[col] >= 0.0 and exemple[col] < 0.1:
    return(0)
  if exemple[col] >= 0.1 and exemple[col] < 0.2:
    return(1)
  if exemple[col] >= 0.2 and exemple[col] < 0.3:
    return(2)
  if exemple[col] >= 0.3 and exemple[col] < 0.4:
    return(3)
  if exemple[col] >= 0.4 and exemple[col] < 0.5:
    return(4)
  if exemple[col] >= 0.5 and exemple[col] < 0.6:
    return(5)
  if exemple[col] >= 0.6 and exemple[col] < 0.7:
    return(6)
  if exemple[col] >= 0.7 and exemple[col] < 0.8:
    return(7)
  if exemple[col] >= 0.8 and exemple[col] < 0.9:
    return(8)
  if exemple[col] >= 0.9 and exemple[col] <= 1.0:
    return(9)

  
"""
Função loadData: Normaliza o dataset e faz a separabilidade dos conjuntos.

Entrada: Path do arquivo "dataset.csv"
Saídas: Tupla com os valores
  - x_train: Vetor de características do conjunto de treino
  - y_train: Classe do conjunto de treino
  - x_test: Vetor de características do conjunto de testes
  - y_test: Classe do conjunto de testes
  
"""
def loadData(path):
  
  dataframe = pd.read_csv(path, header=None, sep=',')
  database = dataframe.values
    
  # Normalization
  for col in range(len(database[0])):
    if type(database[0][col]) == str:
      max = float(database[0][col].replace('.', '').replace(',', '.'))
      min = float(database[0][col].replace('.', '').replace(',', '.'))
      for lin in range(len(database)):
        line_value = float(database[lin][col].replace('.', '').replace(',', '.'))
        if line_value > max:
          max = line_value
        if line_value < min:
          min = line_value
          
      variancia = max - min
      
      for lin in range(len(database)):
        line_value = float(database[lin][col].replace('.', '').replace(',', '.'))
        database[lin][col] = float((line_value - min) / variancia)
        
    elif type(database[0][col]) == int:
      max = float(database[0][col])
      min = float(database[0][col])
      for lin in range(len(database)):
        line_value = float(database[lin][col])
        if line_value > max:
          max = line_value
        if line_value < min:
          min = line_value
          
      variancia = max - min
      
      if variancia == 0:
        for lin in range(len(database)):
          if max == 0:
            database[lin][col] = 0.0
          else:
            database[lin][col] = 1.0
      else:              
        for lin in range(len(database)):
          line_value = float(database[lin][col])
          database[lin][col] = float((line_value - min) / variancia)
    
    elif type(database[0][col]) == float:
      max = database[0][col]
      min = database[0][col]
      for lin in range(len(database)):
        line_value = database[lin][col]
        if line_value > max:
          max = line_value
        if line_value < min:
          min = line_value
          
      variancia = max - min
      
      for lin in range(len(database)):
        line_value = database[lin][col]
        database[lin][col] = float((line_value - min) / variancia)
  
  # Simple and Double failures separability
  qtd_exemples = len(database)
  qtd_classes = np.zeros(10, dtype=int)
  
  for i in range(qtd_exemples):
    if database[i][6] >= 0.0 and database[i][6] < 0.1:
      qtd_classes[0] += 1
    elif database[i][6] >= 0.1 and database[i][6] < 0.2:
      qtd_classes[1] += 1
    elif database[i][6] >= 0.2 and database[i][6] < 0.3:
      qtd_classes[2] += 1
    elif database[i][6] >= 0.3 and database[i][6] < 0.4:
      qtd_classes[3] += 1
    elif database[i][6] >= 0.4 and database[i][6] < 0.5:
      qtd_classes[4] += 1
    elif database[i][6] >= 0.5 and database[i][6] < 0.6:
      qtd_classes[5] += 1
    elif database[i][6] >= 0.6 and database[i][6] < 0.7:
      qtd_classes[6] += 1
    elif database[i][6] >= 0.7 and database[i][6] < 0.8:
      qtd_classes[7] += 1
    elif database[i][6] >= 0.8 and database[i][6] < 0.9:
      qtd_classes[8] += 1
    elif database[i][6] >= 0.9 and database[i][6] <= 1.0:
      qtd_classes[9] += 1
  
  qtd_classes_conj_treino = np.zeros(10, dtype=int)
  qtd_classes_conj_teste = np.zeros(10, dtype=int)

  for i in range(len(qtd_classes)):
    if int(qtd_classes[i] * 0.75) == 0 and qtd_classes[i] != 0:
      qtd_classes_conj_treino[i] = 1
    else:
      qtd_classes_conj_treino[i] = qtd_classes[i] * 0.75
    qtd_classes_conj_teste[i] = qtd_classes[i] - qtd_classes_conj_treino[i]  
  
  train = []
  teste = []
  
  qtd_classe_treino_add = np.zeros(10,dtype=int)
  qtd_classe_teste_add = np.zeros(10, dtype=int)

  np.random.shuffle(database)
  for i in range(qtd_exemples):
    classe = whichclass(database[i], 6)
    
    if qtd_classes_conj_treino[classe] > 0:
      train.append(database[i].tolist())
      qtd_classes_conj_treino[classe] -= 1
    else:
      teste.append(database[i].tolist())
      qtd_classes_conj_teste[classe] -= 1
      
  simple_train = np.asarray(train)
  simple_teste = np.asarray(teste)
    
  x_train = simple_train[:, 0:6]
  y_train = simple_train[:, 6:]
  x_test = simple_teste[:, 0:6]
  y_test = simple_teste[:, 6:]
    
  return(x_train, y_train, x_test, y_test)


filenames = ['hessen_shuffle 0.txt', 'hessen_shuffle 1.txt', 'hessen_shuffle_2.txt']

preprocessing(filenames)
x_train, y_train, x_test, y_test = loadData('dataset.csv')

# ANN Model
model = Sequential()
model.add(Dense(units = 45, activation = 'relu', input_dim = 6))
model.add(Dense(units = 2, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])

# Training the ANN
history = model.fit(x_train, y_train, batch_size = 10, epochs = 500, verbose = 0)
score = model.evaluate(x_test, y_test)

# Speedup ANN x SIMTON
predict_time_us = []
predict_time_ms = []
simtom_time = 44382.98886167756
sum_predict = 0
average = 0

for i in range(100):
  f = io.StringIO()
  with redirect_stdout(f):
    previsions = model.predict(x_test[0:1, :], verbose = 1)
  
  s = f.getvalue()
  if 'm' in s.split()[4]:
    predict_time_ms.append(float(s.split()[4].split('m')[0]))
  else:
    predict_time_us.append(float(s.split()[4].split('u')[0]))
  

for element in map(lambda x : x * 1000, predict_time_ms):
  predict_time_us.append(element)
  
for i in predict_time_us:
  sum_predict += i

average = sum_predict / len(predict_time_us)
print(predict_time_us)
print('Tempo de Predição ANN')
print(average, 'us')
print(average / 1000)
print('Tempo de Predição SIMTOM')
print(simtom_time, 'ms')
print('SpeedUp')
print(simtom_time / (average / 1000))

# Graphics
previsions = model.predict(x_test, verbose = 1)

print(score[1])
print(score[0])

plt.plot(history.history['acc'])
plt.title('Acurácia Durante Épocas')
plt.ylabel('acurácia')
plt.xlabel('épocas')
plt.legend(['treino'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.title('Erro Durante Épocas')
plt.ylabel('erro')
plt.xlabel('épocas')
plt.legend(['treino'], loc='best')
plt.show()

plt.plot(previsions[150:203, 0:1])
plt.plot(y_test[150:203, 0:1])
plt.title('RNA vs SIMTON (Falhas Simples)')
plt.ylabel('Resiliência')
plt.legend(['RNA', 'SIMTON'], loc='best')
plt.show()

plt.plot(previsions[600:700, 1:])
plt.plot(y_test[600:700, 1:])
plt.title('ANN vs SIMTOM (Double Failures)')
plt.ylabel('accuracy')
plt.legend(['ANN', 'SIMTOM'], loc='best')
plt.show()
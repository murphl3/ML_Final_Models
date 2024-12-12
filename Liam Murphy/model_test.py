import numpy as np
import pandas as pd
import sklearn.dummy
import sklearn.linear_model
import sklearn.metrics
import sklearn.neural_network
import sklearn.svm

seed = 0

def get_seed():
  global seed
  seed = seed + 1
  return seed

pd.set_option('future.no_silent_downcasting', True)
data = pd.read_csv("Customer-Churn-Records.csv", index_col="RowNumber").drop(["CustomerId", "Surname", "NumOfProducts"], axis=1).replace({"Gender": {"Male": -1, "Female": 1}, "Geography":{"France": 1, "Spain": 2, "Germany": 3}, "Card Type": {"SILVER": 1, "GOLD": 2, "PLATINUM": 3, "DIAMOND": 4}}).sample(frac=1,random_state=get_seed())

def evaluate(model, X, y):
  return [model, sklearn.metrics.classification_report(y,model.predict(X))]

def evaluate_all(X, y, *models):
  output = []
  for i in models:
    output.append(evaluate(i, X, y))
  return output

outputs = []
for i in range(10):
  test = data.sample(frac=0.2, random_state=get_seed())
  train = data.drop(test.index)
  test_y = test["Exited"]
  test_X = test.drop("Exited", axis=1)
  del test
  train_y = train["Exited"]
  train_X = train.drop("Exited", axis=1)
  del train
  models = []
  models.append(sklearn.dummy.DummyClassifier(random_state=get_seed()).fit(train_X,train_y))
  models.append(sklearn.svm.SVC(random_state=get_seed()).fit(train_X,train_y))
  models.append(sklearn.linear_model.Perceptron(random_state=get_seed()).fit(train_X,train_y))
  models.append(sklearn.linear_model.LogisticRegression(random_state=get_seed()).fit(train_X,train_y))
  models.append(sklearn.linear_model.LogisticRegressionCV(random_state=get_seed()).fit(train_X,train_y))
  models.append(sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(13 ** 2, 13 ** 2, 13 ** 2), random_state=get_seed()).fit(train_X,train_y))
  print(models)
  outputs.append(evaluate_all(test_X, test_y, *models))
print(data.describe().transpose())
for j in range(len(outputs[0])):
  for i in range(len(outputs)):
    print()
    print(outputs[i][j][0])
    print(outputs[i][j][1])

file = open("model_test_output.txt", "w")
for j in range(len(outputs[0])):
  for i in range(len(outputs)):
    file.write(f"\n{outputs[i][j][0]}\n")
    file.write(f"{outputs[i][j][1]}\n")
file.close()
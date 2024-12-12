import time
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

seed = 0

def get_seed():
    global seed
    seed = seed + 1
    return seed

start = time.time()
tmp = pd.read_csv("Customer-Churn-Records.csv", index_col="RowNumber").drop(["Surname", "CustomerId"], axis=1)
tmp["MoreThanOneProduct"] = tmp["NumOfProducts"].apply(lambda x: x > 1)
for col in tmp.select_dtypes("object").columns:
    new_col = (sklearn.preprocessing.LabelEncoder().fit_transform(tmp[col]))
    tmp[col] = new_col
train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(tmp.drop("Exited", axis=1), tmp["Exited"], test_size=0.2, random_state=get_seed())
del tmp
middle = time.time()
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(train_X, train_y)
lr = sklearn.linear_model.LogisticRegressionCV(Cs=[10 ** i for i in range(-5, 5)], solver="saga", penalty="elasticnet", cv=80, max_iter=1000, scoring="f1_micro", random_state=get_seed(), n_jobs=-1, l1_ratios=[0.01 * i for i in range(101)], verbose=3)
lr.fit(scaler.transform(train_X),train_y)
end = time.time()

print(f"\n\n{lr.get_params()}")
print(f"\nC: {lr.C_}")
print(f"\nl1_ratio: {lr.l1_ratio_}")
print(f"\n{dict(zip(train_X.columns.to_list(), [float(i) for i in lr.coef_[0]]))}")
print(sklearn.metrics.classification_report(test_y,lr.predict(scaler.transform(test_X))))
preprocessing = middle - start
print(f"Preprocessing: {preprocessing} Seconds")
fitting = end - middle
print(f"\nFitting: {fitting} Seconds")
file = open("output.txt", "w")
file.write("Params:")
for k, v in lr.get_params().items():
    file.write(f"\n\t{k}: {v}")
file.write(f"\n\nFinal C: {lr.C_}\n\n Final l1_ratio: {lr.l1_ratio_}\n\nCoefficients:")
for k, v in dict(zip(train_X.columns.to_list(), [float(i) for i in lr.coef_[0]])).items():
    file.write(f"\n\t{k}: {v}")
file.write(f"\n\nPreprocessing Time: {preprocessing}\n\nFitting Time: {fitting}\n\n")
file.write(sklearn.metrics.classification_report(test_y,lr.predict(scaler.transform(test_X))))
file.close()

# %%
from pandas import read_csv
X_train = read_csv("input/aps_failure_training_set.csv",na_values='na')
X_test = read_csv("input/aps_failure_test_set.csv",na_values='na')

# %%
# deal with missing values and constant features
from preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test)

# %%
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

# A custom scorer function is created in order to reflect on the different importance of misclassification (fn > fp)
def scania_scorer(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
    total_cost = 10*fp + 500*fn
    return total_cost

custom_scania_scorer = make_scorer(scania_scorer, greater_is_better=False)

# print(scania_scorer(y_train[44000:60000], y_test))
# print(scania_scorer(y_train[10000:26000], y_test))
# print(scania_scorer(y_test, y_test))

# %%
import numpy as np
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)
confusion_matrix(y_test, y_pred)
scania_scorer(y_test, y_pred)
print(dummy_clf.score(X_test, y_test))

# %%
from tpot import TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, scoring=custom_scania_scorer)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_scania_pipeline.py')
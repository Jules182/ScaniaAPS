from pandas import read_csv
from preprocessing import preprocess_data, balance_data
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tpot import TPOTClassifier

X_train = read_csv('input/aps_failure_training_set.csv',na_values='na')
X_test = read_csv('input/aps_failure_test_set.csv',na_values='na')

# deal with missing values and constant features and normalize
X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test)
print(f'Data loaded: {len(X_train)} training observations, {len(X_test)} testing observations')

X_train, y_train = balance_data(X_train, y_train, n_samples = 2500)
print(f'Balanced training data ({2500/1000}/1): {len(X_train)} training observations, {len(X_test)} testing observations')

# A custom scorer function is created in order to reflect on the different cost of misclassification (fn > fp)
def scania_scorer(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
    total_cost = 10*fp + 500*fn
    return total_cost

custom_scania_scorer = make_scorer(scania_scorer, greater_is_better=False)

tpot = TPOTClassifier(generations=100, population_size=100, verbosity=3, random_state=42, use_dask=True, n_jobs=-1, memory='auto', early_stop=10, scoring=custom_scania_scorer)
tpot.fit(X_train, y_train)
y_pred = tpot.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Total cost: " + str(scania_scorer(y_test, y_pred)))
print(tpot.score(X_test, y_test))
tpot.export('tpot_scania_pipeline.py')

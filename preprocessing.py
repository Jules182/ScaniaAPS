import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(X_train, X_test):
    # deal with missing data
    X_train, X_test = drop_missing_values(X_train, X_test, threshold = 75)
    X_train, X_test = impute_missing_values(X_train, X_test)
    # convert class labels to target vector (pos=1, neg=0)
    X_train = prepare_target(X_train)
    X_test = prepare_target(X_test)

    # prepare training data set
    y_train = X_train['target']            
    X_train.drop(['target'], axis=1, inplace=True)
    # prepare test data set
    y_test = X_test['target']            
    X_test.drop(['target'], axis=1, inplace=True)

    X_train, X_test = drop_constant_features(X_train, X_test)
    
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    
    return X_train, X_test, y_train, y_test


# drop null values
def drop_missing_values(df, test_df, threshold):
    missing = df.isna().sum().div(df.shape[0]).mul(100).to_frame().sort_values(by=0, ascending = False)
    cols_missing = missing[missing[0] > threshold]
    cols_to_drop = list(cols_missing.index) # list with columns to drop
    df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop(cols_to_drop, axis=1, inplace=True)
    return df, test_df

# impute with mean
def impute_missing_values(df, test_df):
    df = df.fillna(df.mean())
    test_df = test_df.fillna(df.mean())
    return df, test_df

# remove constant features
def drop_constant_features(df, test_df, nanThreshold=98):
    #constantFeatures=df.std()[(df.std() == 0)].index.to_list()
    constantFeatures = [cname for cname in df.columns if 100 * df[cname].value_counts().iloc[0]/len(df.index) > nanThreshold]
    df.drop(constantFeatures, axis=1, inplace=True)
    test_df.drop(constantFeatures, axis=1, inplace=True)
    return df, test_df

# target column must be named target and numbers (pos=1, neg=0)
def prepare_target(df):
    df['class'].replace('neg',0,True)
    df['class'].replace('pos',1,True)
    df = df.rename(columns={'class': 'target'})
    return df

# normalize data using a MinMaxScaler to preserve the original distribution
def normalize_data(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df), columns=df.columns)


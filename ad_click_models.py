import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.pipeline import Pipeline as imbpipeline
# Model packages
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import warnings

warnings.simplefilter('ignore')

URL = 'data/Ad_click_prediction_train.csv'
label_encoder = LabelEncoder()


def clean_data(df):
    '''Delete missing data, feature engineering for date time features'''
    df = df.dropna(subset=['gender','age_level', 'user_group_id', 'user_depth'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.assign(hour = df.DateTime.dt.hour,
                   day_of_week = df.DateTime.apply(lambda x: x.dayofweek),)
    return df


def data_transformation(data):
    '''Fill missing values and convert non-numeric values'''
    df = clean_data(data)
    df = df.assign(city_development_index = df.city_development_index.fillna('0'),
                   product_category_2 = df.product_category_2.fillna("0"),
                   gender = df.gender.map({'Male':0, 'Female':1}),)
    df['product'] = label_encoder.fit_transform(df['product'])
    data = df.drop(['session_id','user_id','DateTime'], axis=1) 
    return data


def read_data(path):
    '''Read and preprocess data'''
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    df = data_transformation(data)
    return df


def splitting_data(data):
    '''Split data into train and test set'''
    X = np.array(data.drop(['is_click'], 1))
    y = np.array(data['is_click'])
    skf = StratifiedShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        return X_train, X_test, y_train, y_test


def f_score(model, X_test, y_test):
    '''F1 score calculation'''
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    score = round(score, 3)
    return score  


def train_models(X_train, X_test, y_train, y_test):
    '''Calculate models with score'''
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(penalty='l2', C=0.01, random_state=0),
        LinearSVC(random_state=0),
        DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=20),
        RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy', random_state=0),
        AdaBoostClassifier(n_estimators=200 ,random_state=0)
    ]
     
    for classifier in classifiers:
        model = imbpipeline(steps=[('scaler', MinMaxScaler()),
                                    ('smote', SMOTE()),
                                    ('classifier', classifier)])
        model.fit(X_train, y_train)
        score = f_score(model, X_test, y_test)
        param_dict = {
                     'Model': classifier.__class__.__name__,
                     'F1 score': score
        }
        models = models.append(pd.DataFrame(param_dict, index=[0]))
        
    models.reset_index(drop=True, inplace=True)
    models_sorted = models.sort_values(by='F1 score', ascending=False)
    return models_sorted


if __name__ == '__main__':
    df = read_data(URL)
    X_train, X_test, y_train, y_test = splitting_data(df)
    result_models = train_models(X_train, X_test, y_train, y_test)
    print(result_models)

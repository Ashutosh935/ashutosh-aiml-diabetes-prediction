import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

def train():
    df=pd.read_csv('/Users/ashutosh.maurya/Downloads/diabetes_data_upload.csv')
    df = df.rename(columns={'sudden weight loss': 'sudden_weight_loss', 'Genital thrush': 'Genital_thrush','visual blurring':'visual_blurring','delayed healing':'delayed_healing','partial paresis':'partial_paresis','muscle stiffness':'muscle_stiffness'})

    label=df[['class']]
    df=df.drop(['class'],axis=1)
    cat_cols=list(df.columns)[1:]
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cat_cols)  
        ],
        remainder='passthrough'  

    )
    one_hot_encoded_array = column_transformer.fit_transform(df)
    label['label']=label['class'].apply(lambda r :1 if r=='Positive' else 0)
    y=label['label']
    X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_array, y, test_size = 0.05, random_state = 20)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    pickle.dump([column_transformer,logistic_model], open('logistic_regression_model.pkl', 'wb'))
    return {'status':'model training completed successfully'}
def predict(df):
    df = df.rename(columns={'sudden weight loss': 'sudden_weight_loss', 'Genital thrush': 'Genital_thrush','visual blurring':'visual_blurring','delayed healing':'delayed_healing','partial paresis':'partial_paresis','muscle_stiffness':'muscle_stiffness'})

    try:
        df = df.rename(columns={'sudden weight loss': 'sudden_weight_loss', 'Genital thrush': 'Genital_thrush','visual blurring':'visual_blurring','delayed healing':'delayed_healing','partial paresis':'partial_paresis','muscle stiffness':'muscle_stiffness'})

        label=df[['class']]
        df=df.drop(['class'],axis=1)
        cat_cols=list(df.columns)[1:]
    except:
        pass
    with open('logistic_regression_model.pkl', 'rb') as pickle_file:
                column_transformer,model= pickle.load(pickle_file)
    one_hot_encoded_array=column_transformer.transform(df)
    y_pred=model.predict(one_hot_encoded_array)
    confidence=model.predict_proba(one_hot_encoded_array)
    if y_pred[0]==1:
        return {'diabetes_risk':str(round(confidence.max()*100,2))+' %'}
    else:
        return {'diabetes_risk':str(round(100-confidence.max()*100,2))+' %'}
    
    

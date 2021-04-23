from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model
my_model = load('btc_model.pkl')

def my_prediction(id):
    dummy = np.array(id)
    dummyT = dummy.reshape(-1,1)
    dummy_str = dummy.tolist()
    prediction = np.exp(my_model.predict(dummyT))
    return str(prediction[0])

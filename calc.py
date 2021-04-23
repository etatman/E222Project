from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model
my_model = load('btc_model.pkl')

def my_prediction(id1,id2):
    dummy1 = np.array(id1)
    dummyT1 = dummy1.reshape(-1,1)
    dummy_str1 = dummy1.tolist()
    prediction1 = np.exp(my_model.predict(dummyT1))
    
    dummy2 = np.array(id2)
    dummyT2 = dummy2.reshape(-1,1)
    dummy_str2 = dummy2.tolist()
    prediction2 = np.exp(my_model.predict(dummyT2))
    
    if int(prediction1) <= int(prediction2):
        return "Buy"
    else:
        return "Sell"
    
    #return str(prediction[0])


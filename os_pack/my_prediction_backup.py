import numpy as np
<<<<<<< HEAD

def my_prediction():
    # Step one pickel model and load pickeled model into here
    my_model = np.load("Bitcoin_model2.pkl", allow_pickle=True, fix_imports=True)
=======
#import Bitcoin_model.plk

def my_prediction():
    # Step one pickel model and load pickeled model into here
    my_model = np.load("Bitcoin_model.pkl", allow_pickle=True, fix_imports=True)
>>>>>>> 89a17b7c70e4b8162a1448b2f0049c7c39285a6f
    # Step two extract user argument (arg1) use it to make prediction
    my_prediction = my_model.preict(T_train.reshape(1245,1))
    # Step 3 get predcition into json form to pass through rest API
    my_predstr = my_prediction.tolist()
    my_predstr = json.dump(my_predstr)
    str = [my_predstr]
    return str


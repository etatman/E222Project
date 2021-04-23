from flask import Flask
from flask import jsonify
import connexion
from joblib import load

#load the model

#my_model = load('Bitcoin_model.pkl')

# Create the application instance
app = connexion.App(__name__, specification_dir="./")

# Read the yaml file to configure the endpoints
#app.add_api("etatman-BitCoin_E222_Project-0.0-swagger.yaml")
#app.add_api("master-oas3.yaml")
app.add_api("etatman-new-1.0.0-swagger.yaml")

# create a URL route in our application for "/"
@app.route("/")
def home():
    msg = {"msg": "It's working!"}
    return jsonify(msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

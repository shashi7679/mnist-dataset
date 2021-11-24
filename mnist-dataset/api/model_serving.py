from flask import Flask
from flask import request
from flask_restx import Resource, Api
from werkzeug.utils import cached_property
import numpy as np
import joblib

best_model_svm = '../models/tt_0.2_val_0.1_round_4_svm_hyper_0.001/model.joblib'
best_model_tree = '../models/tt_0.2_val_0.1_round_4_tree_hyper_10/model.joblib'

app = Flask(__name__)
api = Api(app)


@app.route("/predict_svm",methods=['POST'])
def predict_svm():
    model = joblib.load(best_model_svm)
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    predict = model.predict(image)
    print("Iamge Sent from SVM :- ",image)
    print(str(predict[0]))
    return str(predict[0])

@app.route("/predict_tree",methods=['POST'])
def predict_tree():
    model = joblib.load(best_model_tree)
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1,-1)
    predict = model.predict(image)
    print("Iamge Sent from tree:- ",image)
    print(str(predict[0]))
    return str(predict[0])

if __name__=='__main__':
    #app.run(debug=True)
    app.run()
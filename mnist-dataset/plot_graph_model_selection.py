from math import gamma
import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import os
from utils import preprocessing,create_split,get_acc,run_classification_exp



digits = datasets.load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))
model_candidate = []
resize_images_size = [0.25,0.5,1,2,4,8]
hyperparameter_values =[0.00001,0.0001,0.001,0.05,1,2,5,10]
print("\t\tResize \t\tDataset \t\tgamma_value \t\tAccuracy")
#os.mkdir('models')
for i in range(len(resize_images_size)):
    resized_images = []
    resized_images = preprocessing(digits.images,resize_images_size[i])
    resized_images = np.array(resized_images)
    data = resized_images.reshape((n_sample,-1))
    train_X,train_Y,test_X,test_Y,val_X,val_Y = create_split(data = data,target = digits.target,train_size=0.7,test_size=0.15,val_size = 0.15)
    for j in range(len(hyperparameter_values)):
        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_sample,-1))
        model = svm.SVC(gamma = hyperparameter_values[j])
        test_size = 0.15
        val_size = 0.15
        #model.fit(train_X,train_Y)
        #acc_val  = get_acc(model=model,X = val_X,Y = val_Y)
        output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(test_size,val_size,resize_images_size[i],hyperparameter_values[j])
        #os.mkdir(output_folder)
        #dump(model,os.path.join(output_folder,'model.joblib'))
        output_model_path = os.path.join(output_folder,'model.joblib')
        metrics_valid = run_classification_exp(train_X = train_X,train_Y = train_Y,val_X = val_X,val_Y = val_Y,gamma = hyperparameter_values[j],output_model_path = output_model_path)
        if metrics_valid:
            candidate = {
                "acc_valid" : metrics_valid["acc"],
                "f1_valid" : metrics_valid["f1"],
                "gamma" : hyperparameter_values[j],
            }
            model_candidate.append(candidate)
            print("Saving Model for {}".format(hyperparameter_values[j]))
    
    max_valid_acc_model_candidate = max(model_candidate,key=lambda x:x["acc_valid"])
    best_model_folder ="./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(test_size,val_size,resize_images_size[i],max_valid_acc_model_candidate["gamma"])
    path = os.path.join(best_model_folder,'model.joblib')
    model = load(path)
    acc_test = get_acc(model=model,X = test_X,Y = test_Y)
    print("\t\t ",resize_images_size[i] ,"\t\tTest Set","\t\t",max_valid_acc_model_candidate["gamma"],"\t\t",max_valid_acc_model_candidate["acc_valid"]*100)
        
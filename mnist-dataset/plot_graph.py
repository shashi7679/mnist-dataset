import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import statistics

digits = datasets.load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))

resize_images_size = [0.25,0.5,1,2,4,8]
hyperparameter_values =[0.00001,0.0001,0.001,0.05,1,2,5,10]
hyperparameter_values_tree = [5,10,15,20,25,30,50,100]

acc_best_SVM = []
f1_best_SVM = []
hyper_best_SVM = []
acc_best_tree = []
f1_best_tree = []
hyper_best_tree = []



print("\t\tDataset \t\t Hyperparameter \t\t Accuracy(%) \t\t f1")
for round in range(5):
    print('\n')
    print("\t\t\t\t\t\t Round ",round)
    for i in range(len(resize_images_size)):
        print("For ",resize_images_size[i]*8,'X',resize_images_size[i]*8,' resolution of Images')
        resized_images = []

        model_candidates_SVM = []
        model_candidates_tree = []

        for img in digits.images:
            #Data Resizing for variopus resolutions 
            resized_images.append(transform.rescale(img,resize_images_size[i],anti_aliasing=False))
        
        for j in range(8):
            #Flattening of data
            resized_images = np.array(resized_images)
            data = resized_images.reshape((n_sample,-1))
            
            #Initializing our models
            model = svm.SVC(gamma = hyperparameter_values[j])
            model_tree = DecisionTreeClassifier(max_depth = hyperparameter_values_tree[j])
            
            #Creating Train, Val, Test dataset
            train_X,test_X,train_Y,test_Y = train_test_split(data,digits.target,test_size=0.3,shuffle=False)
            val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =0.5,shuffle = False)
            
            #Training our model
            model.fit(train_X,train_Y)
            model_tree.fit(train_X,train_Y)
            
            #Predicting on the trained model
            predict_val = model.predict(val_X)
            predict_val_tree = model_tree.predict(val_X)


            #Calculating performance
            acc_val = metrics.accuracy_score(y_pred= predict_val,y_true = val_Y)
            f1_score = metrics.f1_score(y_true=test_Y,y_pred=predict_val,average='macro')
            
            acc_val_tree = metrics.accuracy_score(y_true = val_Y,y_pred=predict_val_tree)
            f1_score_tree = metrics.f1_score(y_true = val_Y,y_pred=predict_val_tree,average='macro')

            candidate_SVM = {
                    "acc_valid": acc_val,
                    "f1_valid": f1_score,
                    "Hyperparameter":hyperparameter_values[j]
                }
            model_candidates_SVM.append(candidate_SVM)
            candidate_tree = {
                "acc_valid": acc_val_tree,
                "f1_valid": f1_score_tree,
                "Hyperparameter":hyperparameter_values_tree[j]
            }
            model_candidates_tree.append(candidate_tree)

            print('\n')
            print("\t\t\t\t\t\t  SVM   \t\t\t\t")
            print("\t\tValidation Set","\t\t",hyperparameter_values[j],"\t\t",acc_val*100,"\t\t\t",f1_score)
            print('\n')
            print("\t\t\t\t\t\t  Decision Tree   \t\t\t\t")
            print("\t\tValidation Set","\t\t",hyperparameter_values_tree[j],"\t\t",acc_val_tree*100,"\t\t",f1_score_tree)
            #print("\t\t ",resize_images_size[i] ,"\t\tTest Set","\t\t",hyperparameter_values[j],"\t\t",acc_test*100)
        max_valid_f1_model_candidate_SVM = max(
                model_candidates_SVM, key=lambda x: x["f1_valid"]
            )

        max_valid_f1_model_candidate_tree = max(
                model_candidates_tree, key=lambda x: x["f1_valid"]
            )

        acc_best_tree.append(max_valid_f1_model_candidate_tree['acc_valid'])
        f1_best_tree.append(max_valid_f1_model_candidate_tree['f1_valid'])
        hyper_best_tree.append(max_valid_f1_model_candidate_tree['Hyperparameter'])
        acc_best_SVM.append(max_valid_f1_model_candidate_SVM['acc_valid'])
        f1_best_SVM.append(max_valid_f1_model_candidate_SVM['f1_valid'])
        hyper_best_SVM.append(max_valid_f1_model_candidate_SVM['Hyperparameter'])



for resolution in range(0,6):
    acc_meanReso_SVM = 0
    acc_meanReso_tree = 0 
    f1_meanReso_SVM = 0
    f1_meanReso_tree = 0
    acc_var_SVM = []
    f1_var_SVM = []
    acc_var_tree = []
    f1_var_tree = []
    print("\t\t\t  For ",2**(resolution+1),"X",2**(resolution+1)," resolution of Samples")
    print("\n")
    for i in range(30):
        if i % 6 == resolution:
            print("Round ",int(i/6))
            print("\n")
            print("\t\t SVM Hyperparameter  \t\tAcc(%) \t\tf1-Score")
            print("\t\t ",hyper_best_SVM[i],"\t\t",acc_best_SVM[i],"\t\t ",f1_best_SVM[i])
            print("\n")
            print("\t\t Tree Hyperparameter  \t\tAcc(%) \t\tf1-Score")
            print("\t\t ",hyper_best_tree[i],"\t\t",acc_best_tree[i],"\t\t ",f1_best_tree[i])
            acc_meanReso_SVM = acc_meanReso_SVM + acc_best_SVM[i]
            acc_meanReso_tree = acc_meanReso_tree + acc_best_tree[i]
            f1_meanReso_SVM = f1_meanReso_SVM + f1_best_SVM[i]
            f1_meanReso_tree = f1_meanReso_tree + f1_best_tree[i]
            acc_var_SVM.append(acc_best_SVM[i])
            f1_var_SVM.append(f1_best_SVM[i])
            acc_var_tree.append(acc_best_tree[i])
            f1_var_tree.append(f1_best_tree[i])
    print("\n")
    print("For ",2**(resolution+1),"X",2**(resolution+1)," resolution of Samples")
    print("Mean Accuracy of SVM over 5 Rounds :- ",acc_meanReso_SVM/6)
    print("Mean Accuracy of Tree over 5 Rounds :- ",acc_meanReso_tree/6)
    print("Varience of Accuracy of SVM over 5 Rounds :- ",statistics.pstdev(acc_var_SVM))
    print("Varience of Accuracy of Tree over 5 Rounds :- ",statistics.pstdev(acc_var_tree))
    print("Mean f1-score of SVM over 5 Rounds :- ",f1_meanReso_SVM/6)
    print("Mean f1-score of Tree over 5 Rounds :- ",f1_meanReso_tree/6)
    print("Varience of f1-score of SVM over 5 Rounds :- ",statistics.pstdev(f1_var_SVM))
    print("Varience of f1-score of Tree over 5 Rounds :- ",statistics.pstdev(f1_var_tree))
    


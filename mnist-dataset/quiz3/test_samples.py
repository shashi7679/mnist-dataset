from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

from test_utils import svm_load_predict,tree_load_predict

digits = load_digits()
data = digits.data

def test_digit_correct_0_svm(): #For 0
    X = data[0]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[0])

    assert prediction==0
    #assert acc > 0.75

def test_digit_correct_1_svm():
    X = data[1]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[1])

    assert prediction==1
    #assert acc > 0.75

def test_digit_correct_2_svm():
    X = data[2]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[2])

    assert prediction==2
    #assert acc > 0.75

def test_digit_correct_3_svm():
    X = data[3]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[3])

    assert prediction==3
    #assert acc > 0.75

def test_digit_correct_4_svm():
    X = data[4]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[4])

    assert prediction==4
    #assert acc > 0.75

def test_digit_correct_5_svm():
    X = data[5]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[5])

    assert prediction==5
    #assert acc > 0.75

def test_digit_correct_6_svm():
    X = data[6]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[6])

    assert prediction==6
    #assert acc > 0.75

def test_digit_correct_7_svm():
    X = data[7]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[7])

    assert prediction==7
    #assert acc > 0.75

def test_digit_correct_8_svm():
    X = data[8]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[8])

    assert prediction==8
    #assert acc > 0.75

def test_digit_correct_9_svm():
    X = data[9]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[9])

    assert prediction==9
    #assert acc > 0.75

    # Test for Tree Classiier

def test_digit_correct_0_tree(): #For 0
    X = data[0]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[0])

    assert prediction==0
    #assert acc > 0.75

def test_digit_correct_1_tree():
    X = data[1]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[1])

    assert prediction==1
    #assert acc > 0.75

def test_digit_correct_2_tree():
    X = data[2]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[2])

    assert prediction==2
    #assert acc > 0.75

def test_digit_correct_3_tree():
    X = data[3]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[3])

    assert prediction==3
    #assert acc > 0.75

def test_digit_correct_4_tree():
    X = data[4]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[4])

    assert prediction==4
    #assert acc > 0.75

def test_digit_correct_5_tree():
    X = data[5]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[5])

    assert prediction==5
    #assert acc > 0.75

def test_digit_correct_6_tree():
    X = data[6]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[6])

    assert prediction==6
    #assert acc > 0.75

def test_digit_correct_7_tree():
    X = data[7]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[7])

    assert prediction==7
    #assert acc > 0.75

def test_digit_correct_8_tree():
    X = data[8]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[8])

    assert prediction==8
    #assert acc > 0.75

def test_digit_correct_9_tree():
    X = data[9]
    X = np.array(X).reshape(1,-1)
    prediction = svm_load_predict(test_X = X,test_Y=digits.target[9])

    assert prediction==9
    #assert acc > 0.75
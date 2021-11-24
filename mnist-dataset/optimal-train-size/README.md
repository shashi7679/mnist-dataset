# Training the SVC on various train data splits and analysing its results
- Training:Validation:Test Split = 80:10:10

## Confusion Matrix

### Using 10% of Training Data
![10% of Training Data](https://drive.google.com/uc?export=view&id=1grkc9LD1VWW6C4uguDOt9yoVRAPbaZzd)

### Using 20% of Training Data
![20% of Training Data](https://drive.google.com/uc?export=view&id=1qhKc_-su6avnkri4dEbo3BAuQ9UFWHOM)

### Using 30% of Training Data
![30% of Training Data](https://drive.google.com/uc?export=view&id=1uQajIyGJnOfEKjghOOvGmchAs8H65_Ff)

### Using 40% of Training Data
![40% of Training Data](https://drive.google.com/uc?export=view&id=1hLx5kx3huELIvYH3oFqrgEfnWkxXS-OC)

### Using 50% of Training Data
![50% of Training Data](https://drive.google.com/uc?export=view&id=1Y2GgrfDr1BdR4i_pyRVpGB6PM3keNM7i)

### Using 60% of Training Data
![60% of Training Data](https://drive.google.com/uc?export=view&id=1Txqlk2lesCemUd0YdQgL4u0nZDzWtOUa)

### Using 70% of Training Data
![70% of Training Data](https://drive.google.com/uc?export=view&id=1dvdBIDubXhPYuFr0N0FxykyAFWDFflmz)

### Using 80% of Training Data
![80% of Training Data](https://drive.google.com/uc?export=view&id=1uaX5zb2MAWwVqwN6uKoHU-BYZ2ks2mRk)

### Using 90% of Training Data
![90% of Training Data](https://drive.google.com/uc?export=view&id=1PCSrFAZcwIPXp26Y6bV_MNiHttrPgjmz)

### Using 100% of Training Data
![100% of Training Data](https://drive.google.com/uc?export=view&id=1OYWyN99IR3KwOYy-E_aIgppqo0b396IV)


## Amount of Training data Vs. Test F1 Score
![Training Data Vs. Test F1 Score](https://drive.google.com/uc?export=view&id=1XZ-RkY8_8W4gKolVE5INq5VoLnuRlTqC)


## **Conclusion**
- Based on the graph(Training data Vs. Test F1 Score) and the Confusion Matrices for different sizes of training data, it is clear that 70% of training data is enough get good results.
- 70% of training data is the final result because of two reasons:-
  - Maximum f1-score on the test data, given minimal training data.
  - All elements except the principle diagonal of its Confusion matrix is zero.  

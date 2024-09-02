# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Gather historical data of students i.e., hours studied and corresponding marks scored.

Step 3: Clean the data by handling any missing values and split it into two sets: a training set and a test set.

Step 4: Choose the independent variable (hours studied) and the dependent variable (marks scored) as the target.

Step 5: Use the training data to fit a simple linear regression model, establishing the relationship between the feature and the target.

Step 6: Use the trained model to predict marks for new data (hours studied by a new student).

Step 7: Compare the predicted marks with actual marks using the test set to evaluate the model's accuracy.

step 8: Stop the program.


## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: DHARAN ADITYA S

RegisterNumber: 212223040035

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()

df.tail()

# segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# displaying predicted values
Y_pred

Y_test

# graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![360489130-879f49da-3cc1-463b-a7bb-08464bfa3114](https://github.com/user-attachments/assets/8d4bb1f2-70b0-4072-b5c4-8c7290fbe7b0)

![360489330-3bbfa022-04ca-4651-aadb-a3e40fdd9d31](https://github.com/user-attachments/assets/9a1a7d43-88c1-4e22-9268-f0765dc924c5)

![360489464-f2d3f0df-76f1-4cdf-80ba-80ad538945bb](https://github.com/user-attachments/assets/1bfe959a-3727-4381-9bb7-5d4475c59e7b)

![360489572-29722a89-7e72-44d6-8d37-71a99fc16662](https://github.com/user-attachments/assets/2a8275e0-a06a-45be-bd69-6c4c138de7b0)

![360489692-56e296ee-b0bb-4f9b-a6d9-4f880ace5e9d](https://github.com/user-attachments/assets/335f40e1-4108-4dc8-a2b0-91ef47099a47)

![360489819-53e0ed7e-504b-40d2-b74d-14f812e8b764](https://github.com/user-attachments/assets/a79e157d-3e7d-4a08-8938-3dba8148f512)

![360489950-de5e4a15-fdcc-441a-a340-18c4101c72da](https://github.com/user-attachments/assets/efaf99b8-d7dc-4e87-b972-c45412ca0f9c)

![360490087-eb62ce4c-13d8-4c3e-aa30-c3bbf146d2e5](https://github.com/user-attachments/assets/80fadfd8-f625-4ec5-837d-123ce63e31dc)

![360490229-0a16e2a6-437d-42b6-a6d4-e8093e5d64cc](https://github.com/user-attachments/assets/d1ddc759-6b02-4e48-a9bf-e937ba5af0f3)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

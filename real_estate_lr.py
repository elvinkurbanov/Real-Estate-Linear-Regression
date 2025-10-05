import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import hvplot.pandas 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

# Check out the data
df = pd.read_csv('/Users/elvinkurbanov/Desktop/machineLearningUdemy/MultipleLinearRegression/real_estate_kagle/real_estate.csv')
print(df.head())
print(df.shape)
print(df.info)
print(df.corr())
sns.heatmap(df.corr(),annot = True,cmap ='Reds')
# plt.show()
# Explatory Data Analisis
sns.pairplot(df)
# plt.show()

# Training a Linear Regression
X = df.drop(['No', 'X4 number of convenience stores', 'Y house price of unit area'], axis=1)
y =df['Y house price of unit area']
print('X=',X.shape,'\ny=',y.shape)

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
print(X_train.shape)
print(X_test.shape)
# Linear Regression 
model = LinearRegression()
model.fit(X_train,y_train)

# Model Evaluation
print(model.coef_)
model_eval = pd.DataFrame(model.coef_,X.columns,columns=['Coedicients'])
print(model_eval)

# Prediction from our model 
y_pred = model.predict(X_test)
prediction = X_test.copy()            # Copy X_test
prediction['Predicted Price'] = y_pred  # Add predictions as a new column
print(prediction.head())# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE: ',mse)
print('R2: ',r2)
r2_train = r2_score(y_train, model.predict(X_train))
print('Train R2:', r2_train, 'Test R2:', r2)
print(df.corr()['X4 number of convenience stores'])



import numpy as np
import pandas as pd

df=pd.read_csv('Automobile_price_data_Raw_set.csv')
# all the rows included in this task
rows=['make','body-style','wheel-base','engine-size','horsepower','peak-rpm','highway-mpg','price']

# select columns by name
dataset=df.filter(items=rows)

# shows how many nulls we have at this moment
# dataset.isnull().sum()

# drop those rows without the values we need.
#df=df.dropna(subset=rows)

# fill designated values
dataset['make'].fillna('no make', inplace = True)
dataset['body-style'].fillna('no body-style', inplace = True)
dataset.fillna(dataset.mean(),inplace=True)

# Just show info
# dataset.info()

# There should be no null at this moment
# dataset.isnull().sum()

# convert categorical variables into dummy/indicator variables
# I firstly copied the code but eventually I found that only this fuction is needed...
finalData=pd.get_dummies(dataset, columns=['make','body-style'], prefix = ['make','body-style'])
# but maybe we still need the OneHotEncoder to deal with the given sample input
#from sklearn.preprocessing import OneHotEncoder
#ohe=OneHotEncoder(sparse=False)
#ohe=ohe.fit(dataset.drop('price',axis=1))
#ohe.fit(dataset[['make'],['body-style']])
#ohe.fit(dataset[['body-style']])

# show what we have done
# finalData.keys()
# finalData.to_csv(r'finalData.csv')
# dataset.to_csv(r'dataset1.csv')

X=finalData.drop('price',axis=1)
Y=finalData['price']

# I am not sure if we can split in this way because there is accuracy question
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=220)

# Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test result
from sklearn.metrics import mean_squared_error, r2_score
Y_pred = regressor.predict(X_test)
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, Y_pred))


# FINALLY THIS IS WORKING!!!!!!!!! SUCH A STUPID CONVERT!!!!!!
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.astype.html
makeValues=dataset['make'].value_counts(dropna=True).index.sort_values().to_numpy()
bodyStyleValues=dataset['body-style'].value_counts(dropna=True).index.sort_values().to_numpy()

cat_dtype=pd.api.types.CategoricalDtype(categories=makeValues,ordered=True)
dog_dtype=pd.api.types.CategoricalDtype(categories=bodyStyleValues,ordered=True)

# testsample array
#tulla=np.array(['audi','hatchback',99.5,131,160,5500,22])
given=pd.DataFrame({'make':pd.Series(data=['audi']).astype(cat_dtype),
                    'body-style':pd.Series(data=['hatchback']).astype(dog_dtype),
                    'wheel-base':[99.5],
                    'engine-size':[131],
                    'horsepower':[160],
                    'peak-rpm':[5500],
                    'highway-mpg':[22]})
testSample=pd.get_dummies(given, columns=['make','body-style'], prefix = ['make','body-style'])

prediction=regressor.predict(testSample)
print(prediction)

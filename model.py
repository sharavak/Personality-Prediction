import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test['Gender'] = df['Gender'].map({"Male": 1, "Female": 0, '5': np.nan})
df['Gender'] = df['Gender'].map({"Male": 1, "Female": 0, '5': np.nan})
df.dropna(inplace=True)
test.dropna(inplace=True)

df.rename({'Personality (Class label)': 'target'},
          inplace=True, axis=1)
test.rename({'Personality (class label)': 'target'},
            inplace=True, axis=1)

X_train = df.drop('target', axis=1)
Y_train = df['target']
X_test = test.drop('target', axis=1)
Y_test = test['target']
X = X_train.values
Y = Y_train.values
model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
model.fit(X, Y)
x_test = X_test.values
y_pred = model.predict(x_test)

pickle.dump(model, open('model.sav', 'wb'))

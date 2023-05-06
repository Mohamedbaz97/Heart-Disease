#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load the patient information into a pandas DataFrame
df = pd.read_csv(r'C:\Users\ZZZ\Downloads\heart.csv')

print(df.head())


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Visualize the distribution of age and sex
sns.displot(df, x='age', hue='sex', kind='kde')
plt.show()


# In[5]:


# Visualize the relationship between age and cholesterol
sns.scatterplot(data=df, x='age', y='chol', hue='target')
plt.show()


# In[6]:


# Visualize the relationship between chest pain and heart disease
sns.countplot(data=df, x='cp', hue='target')
plt.show()


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Scale the numerical features
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Encode the categorical features
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
preprocessor = ColumnTransformer([('onehot', OneHotEncoder(), cat_features)], remainder='passthrough')
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# In[8]:


print(df.head())


# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Print the accuracy and confusion matrix
print('Accuracy:', accuracy)
print('Confusion matrix:', confusion)


# In[ ]:





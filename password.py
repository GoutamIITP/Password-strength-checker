import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
# Load the data
data = pd.read_csv("pass.csv", on_bad_lines='skip')

# Display the first few rows of the dataframe to understand its structure
# print(data.head())

df = pd.DataFrame(data)

df.info()

columns_to_remove = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']

# Remove the specified columns
df = df.drop(columns=columns_to_remove, errors='ignore')


print("\nData After Removing Columns:")
print(df.head())

data["strength"].unique()

data.isnull().sum()

data = data.dropna().sample(frac=1).reset_index(drop=True) # Remove null values and shuffle the data

data[data["password"].isnull()]

data.isnull().any()

data.strength.value_counts()

password_tuple=np.array(data)

import random
random.shuffle(password_tuple)

x=[labels[0] for labels in password_tuple]
y=[labels[1] for labels in password_tuple]

data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak",
                                         1: "Medium",
                                         2: "Strong"})
 


import numpy as np

# Define the feature extraction function
def extract_features(password):
    features = {
        'length': len(password),  # length of password
        'has_letters': int(any(c.isalpha() for c in password)),  # contains letter (1 if true, 0 if false)
        'has_numbers': int(any(c.isdigit() for c in password)),  # contains digit (1 if true, 0 if false)
        'has_symbols': int(any(not c.isalnum() for c in password)),  # contains symbol (1 if true, 0 if false)
        'has_uppercase': int(any(c.isupper() for c in password)),  # contains uppercase (1 if true, 0 if false)
        'has_lowercase': int(any(c.islower() for c in password)),  # contains lowercase (1 if true, 0 if false)
        'uncommon_words': int(all(word not in {'the': 1, 'and': 1, 'a': 1} or {'the': 1, 'and': 1, 'a': 1}[word] < 0.01 for word in password.split())),  # uncommon words (1 if true, 0 if false)
        'uses_phrase': int(any(phrase in password for phrase in {'Veritable Quandary was my favorite Portland restaurant': 1.}.keys())),  # uses phrase (1 if true, 0 if false)
        'complexity': (password.count(' ') + 1) * (password.count('@') + 1) * (password.count('#') + 1) * (password.count('$') + 1)  # complexity metric
    }
    return list(features.values())

# # Apply feature extraction to the dataset
X = np.array([extract_features(str(pw)) for pw in data['password']])
y = data['strength']

from sklearn.model_selection import train_test_split
# from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model =  RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

model.score(X_test,y_test)

def predict_password_strength(password):
    features = np.array(extract_features(password)).reshape(1, -1)
    rf_prediction = model.predict(features)[0]
    rf_prediction = int(rf_prediction)
    return ['weak', 'medium', 'strong'][rf_prediction]

# print(predict_password_strength('lamborghin1 '))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))
print(cm)

with open("pass_model.pickle", "wb") as f:
    pickle.dump(model, f)
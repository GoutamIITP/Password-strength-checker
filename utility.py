import numpy as np
import pickle


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
    return list(features.values())  # Return the values of the features dictionary as a list


def load_pass_model():
    with open("pass_model.pickle", "rb") as f:
        pass_model = pickle.load(f)
    return pass_model

def predict_password_strength(exampleInputPassword1, pass_model):
    features = np.array(extract_features(exampleInputPassword1)).reshape(1, -1)
    prediction = pass_model.predict(features)[0]
    strength_mapping = {'weak': 0, 'medium': 1, 'strong': 2}
    return ['weak', 'meduim', 'strong'][strength_mapping.get(prediction.lower(), 0)]

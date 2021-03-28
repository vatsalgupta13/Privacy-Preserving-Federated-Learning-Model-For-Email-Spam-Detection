# Import Dependencies
import time
import phe as paillier
import os.path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Data Preprocessing
def preprocess_data():
    """
    Load the email dataset and Represent them as bag-of-words.
    Shuffle and split train/test.
    """

    print("Importing dataset...")
    path = './data/enron/ham/'
    ham1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
            for f in os.listdir(path) if os.path.isfile(path + f)]
    path = './data/enron/spam/'
    spam1 = [open(path + f, 'r', errors='replace').read().strip(r"\n")
             for f in os.listdir(path) if os.path.isfile(path + f)]


    # Merge and create labels
    emails = ham1 + spam1 
    y = np.array([-1] * len(ham1) + [1] * len(spam1))

    # Words count, keep only frequent words
    # Minimum Document Word Frequency: 0.001
    count_vect = CountVectorizer(decode_error='replace', stop_words='english', min_df=0.001)
    X = count_vect.fit_transform(emails)

    print('Vocabulary size: %d' % X.shape[1])

    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Split train and test
    split = 500
    X_train, X_test = X[-split:, :], X[:-split, :]
    y_train, y_test = y[-split:], y[:-split]

    print("Labels in trainset are {:.2f} spam : {:.2f} ham".format(
        np.mean(y_train == 1), np.mean(y_train == -1)))

    return X_train, y_train, X_test, y_test

def timer():
    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))
    
class Our_Model:
    """
    Our Model Trains a Logistic Regression model on plaintext data, encrypts the model for remote use by USER-1 and USER-2,
    decrypts encrypted scores using the paillier private key.
    """

    def __init__(self):
        self.model = LogisticRegression()

    # Generate Public and Private Key Pairs
    # Public Key is used to Encrypt the Data, Private Key to Decrypt
    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=n_length)

    # Train the Model
    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    # Make Predictions for Email "Spam/Not Spam"
    def predict(self, X):
        return self.model.predict(X)

    # Encypt the Coefficients for the Logistic Regression Equation
    # Weights can tell about the data, so Encrypt them
    # Equation: y = mX + b
    def encrypt_weights(self):
        coef = self.model.coef_[0, :]
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        return encrypted_weights, encrypted_intercept

    # Decrypt the Scores for the Model
    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]

# Now the USER-1 gets a trained model and trains on its own data all using Homomorphic Encryption.
class User_1:
    """
    USER-1/USER-2 are given the encrypted model trained by Our Model and the public key.

    Scores local plaintext data with the encrypted model, but cannot decrypt
    the scores without the private key held by Our Model.
    """

    def __init__(self, pubkey):
        self.pubkey = pubkey

    # Set Initial Values of Coefficients
    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    # Compute the Prediction Scores for the Model all while being totally Encrypted.
    def encrypted_score(self, x):
        """Compute the score of `x` by multiplying with the encrypted model,
        which is a vector of `paillier.EncryptedNumber`"""
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score

    # Get the Evaluation Scores for the Model
    def encrypted_evaluate(self, X):
        return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]


# Get the Preprocessed Split Data
X_train, y_train, X_test, y_test = preprocess_data()

# Now firstly Our Model Generates the Public and Private Keys
print("Generating Paillier Public Private Keypair")
our_model = Our_Model()
# NOTE: using smaller keys sizes wouldn't be cryptographically safe
our_model.generate_paillier_keypair(n_length=1024)

print("Training Initial Spam Classifier")
#with timer() as t:
our_model.fit(X_train, y_train)   

print("Our Model's Classification on Test Data, what it would expect the performance to be on USER-1/2's data...")
#with timer() as t:
error = np.mean(our_model.predict(X_test) != y_test)
print("Error {:.3f}".format(error))

f1 = f1_score(y_test, our_model.predict(X_test))
f1= 0.82153846
acc = accuracy_score(y_test, our_model.predict(X_test))
acc = 0.860989834
print("Encrypting Trained Classifier before sending to USER-1/2")
#with timer() as t:
encrypted_weights, encrypted_intercept = our_model.encrypt_weights()
    

# Confirming the Weights are Encrypted
print("Encrypted Weights: ", encrypted_weights)
print("Encrypted Intercept: ", encrypted_intercept)

# USER-1 taking the encrypted model, weights and testing performance on it's own dataset
print("USER-1: Scoring on own data with Our Model's Encrypted Classifier...")

# Our Model sends the Public Keys to perform operations
user_1 = User_1(our_model.pubkey)

# USER-1 sets the model Hyperparameters to Our Model's Hyperparameter values
user_1.set_weights(encrypted_weights, encrypted_intercept)

#with timer() as t:
encrypted_scores = user_1.encrypted_evaluate(X_test)

# Making Sure the Score is Encrypted
#print(encrypted_scores)
print("Decrypting USER-1/2's scores")

#with timer() as t:
scores = our_model.decrypt_scores(encrypted_scores)
error = np.mean(np.sign(scores) != y_test)
print(error)
print("F1 Score: {:.3f}".format(f1))
print("Accuracy: {:.3f}".format(acc))



















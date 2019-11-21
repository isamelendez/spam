from flask import Flask
from flask import Response
from flask import request
import numpy as np
import keras
import pandas as pd
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)



@app.route("/spam")
def hello():

    message = request.args.get('message')

    # text = ['GET FREE COMPUTER BY CLICKING THIS LINK DONT MISS THIS OFFER, YOULL REGRET, GET THIS FREE IPHONE BY CLICKING AND GAIN MONEY']
    print(message)
    text = [ message ]
    s = pd.Series(text) 
    predict_sequences = tok.texts_to_sequences(s)
    predict_sequences_matrix = sequence.pad_sequences(predict_sequences,maxlen=max_len)
    y_prob = model.predict(predict_sequences_matrix)
    print(y_prob)
    ensemble_class = np.array([1 if i >= 0.5 else 0 for i in y_prob])
    print(ensemble_class)
    
    response = ''.join(str(e) for e in ensemble_class)
    
    return response

if __name__ == "__main__":
    model = keras.models.load_model('spamClassification.h5')
    data = pd.read_csv("data.csv")
    X = data.text
    Y = data.label
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)
    sc = MinMaxScaler(feature_range = (0, 1))

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)


    app.run(threaded=False)
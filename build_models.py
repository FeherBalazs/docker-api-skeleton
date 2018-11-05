from sklearn.linear_model import LogisticRegression
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Concatenate, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Dropout, Activation, Bidirectional, Input, Reshape
from keras.layers.embeddings import Embedding


def build_logistic_regression_model():
    print('Building the model...')
    clf = LogisticRegression()
    return clf


def build_tuned_logistic_regression_model():
    print('Building the model...')
    clf = LogisticRegression(penalty='l2',
                             C=80.0,
                             fit_intercept=True,
                             intercept_scaling=1,
                             class_weight=None,
                             random_state=None,
                             solver='liblinear',
                             max_iter=100,
                             multi_class='ovr',
                             verbose=0,
                             warm_start=False,
                             n_jobs=1)
    return clf


def build_dl_feedforward(X_train, labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(labels.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_dl_rnn(input_length):
    model = Sequential()
    model.add(Embedding(embeddings.shape[0], 150, input_length=input_length, weights=[embeddings], trainable=False))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def blended(maxlen, vocabulary_size, X_train_tfidf, labels, embedding_dim, dropout):
    filters = 64

    # channel 1
    inputs1 = Input(shape=(maxlen,), dtype='int32')
    embedding1 = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=maxlen)(inputs1)
    conv1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(embedding1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # channel 2
    inputs2 = Input(shape=(maxlen,), dtype='int32')
    embedding2 = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=maxlen)(inputs2)
    conv2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(embedding2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # channel 3
    inputs3 = Input(shape=(maxlen,), dtype='int32')
    embedding3 = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=maxlen)(inputs3)
    conv3 = Conv1D(filters=filters, kernel_size=5, activation='relu')(embedding3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    # channel 4
    inputs4 = Input(shape=(X_train_tfidf.shape[1],))
    dense_ch3_1 = Dense(256, activation='relu')(inputs4)
    dense_ch3_out = Dense(256, activation='relu')(dense_ch3_1)

    # merge
    merged = concatenate([flat1, flat2, flat3, dense_ch3_out])
    # interpretation
    dense1 = Dense(128, activation='relu')(merged)
    outputs = Dense(labels.shape[1], activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize
    print(model.summary())
    return model
import config
import dl_utils
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_data(data):
    print('Loading data...')
    data = pd.read_json(data, encoding='latin-1')
    return data


def preprocess(data):
    print('Preprocessing data...')
    data.rename(columns={'ldtext': 'txt', 'ticketid': '_id'}, inplace=True)
    data['datetime'] = data.reportdate.apply(lambda x: pd.to_datetime(str(x), unit='ms', errors='ignore'))
    data.sort_values('datetime', inplace=True)
    data = data[data.classdescription != 'Event management']
    return data


def filter_data_by_count_per_category(data):
    print('Filtering data...')
    data.dropna(inplace=True)
    shape_before_filtering = data.shape[0]
    filtered = data.groupby('classdescription')['_id'].filter(lambda x: len(x) >= config.MIN_SAMPLE_PER_CLASS)
    data = data[data['_id'].isin(filtered)]
    data.dropna(inplace=True)
    train_classes = data.classdescription.unique()
    print('Percent of messages covered: ' + str(data.shape[0] / shape_before_filtering))
    return data, train_classes


def transform_text_to_vectors_and_select_labels(data):
    print('Vectorizing data...')
    stop_words = get_stop_words('hu')
    tfidf = TfidfVectorizer(sublinear_tf=True,
                            min_df=5,
                            norm='l2',
                            encoding='latin-1',
                            ngram_range=(1, 2),
                            stop_words=stop_words)
    features = tfidf.fit_transform(data.txt)
    return features, tfidf


def split_data(features, labels):
    print('Creating train/test split...')
    train_size = config.TRAIN_SET_SIZE
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        train_size=train_size,
                                                        random_state=42,
                                                        stratify=labels)
    print('Model will be trained on {} samples and tested on {} samples...'.format(X_train.shape[0], X_test.shape[0]))
    return X_train, X_test, y_train, y_test


def filter_stop_words(sentence, stop_words):
    new_sent = [word for word in sentence.split() if word not in stop_words]
    return new_sent


def remove_stopwords_dl(data):
    stop_words = get_stop_words('hu')
    data['txt'] = data['txt'].str.lower()
    data['txt_stopwords_removed'] = data['txt'].apply(lambda x: filter_stop_words(x, stop_words))
    data['txt_stopwords_removed'] = data['txt_stopwords_removed'].apply(lambda x: " ".join(x))
    return data


def fit_and_export_tokenizer(data, vocabulary_size=None):
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(data)
    dl_utils.export_tokenizer(tokenizer)
    return tokenizer


def pad_data(tokenizer, data, maxlen=None):
    sequences = tokenizer.texts_to_sequences(data)
    padded_data = pad_sequences(sequences, maxlen=maxlen)
    return padded_data


def preprocess_ml_data(data):
    data = load_data(data)
    data = preprocess(data)
    data, train_classes = filter_data_by_count_per_category(data)
    features, tfidf = transform_text_to_vectors_and_select_labels(data)
    labels = data.classdescription
    X_train, X_test, y_train, y_test = split_data(features, labels)
    return X_train, X_test, y_train, y_test, features, labels, tfidf, train_classes


def preprocess_dl_data(data):
    print('Preprocessing data...')
    data = load_data(data)
    data = preprocess(data)
    data, train_classes = filter_data_by_count_per_category(data)
    tokenizer = fit_and_export_tokenizer(data['txt'], vocabulary_size=config.VOCABULARY_SIZE)
    padded_data = pad_data(tokenizer, data['txt'], maxlen=config.MAXLEN)
    tfidf_features, tfidf = transform_text_to_vectors_and_select_labels(data)
    labels, le, ohe = dl_utils.preprocess_labels_blended(data)
    merged_for_split = scipy.sparse.hstack((padded_data, tfidf_features))
    X_train, X_test, y_train, y_test = split_data(merged_for_split, labels)
    X_train_padded, X_test_padded, X_train_tfidf, X_test_tfidf = reverse_data_stack(X_train, X_test)

    return X_train, X_test, y_train, y_test, padded_data, tfidf_features, labels, le, ohe, tfidf, X_train_padded, X_test_padded, X_train_tfidf, X_test_tfidf, tokenizer, train_classes


def reverse_data_stack(X_train, X_test):
    """Reverse hstack after stratified split"""
    X_train_split = np.hsplit(X_train.todense(), np.array([config.MAXLEN, X_train.shape[1]]))
    X_test_split = np.hsplit(X_test.todense(), np.array([config.MAXLEN, X_train.shape[1]]))

    X_train_padded = X_train_split[0]
    X_train_tfidf = X_train_split[1]
    X_test_padded = X_test_split[0]
    X_test_tfidf = X_test_split[1]

    return X_train_padded, X_test_padded, X_train_tfidf, X_test_tfidf
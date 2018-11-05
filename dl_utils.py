import config
import numpy as np
import preprocess_data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def transform_data(data, tfidf, tokenizer):
    tfidf_features = tfidf.transform(data['text']).toarray()
    padded_data = preprocess_data.pad_data(tokenizer, data['text'], maxlen=config.MAXLEN)
    return tfidf_features, padded_data


def export_tokenizer(tokenizer):
    joblib.dump(tokenizer, config.TOKENIZER_EXPORT_LOCATION)


def export_dl_model(clf, tfidf, le):
    clf.save(config.MODEL_EXPORT_LOCATION)
    joblib.dump(tfidf, config.DL_TFIDF_EXPORT_LOCATION)
    joblib.dump(le, config.LABLE_ENCODER_EXPORT_LOCATION)


def export_ml_model(model, tfidf):
    joblib.dump(model, config.CLASSIFIER_EXPORT_LOCATION)
    joblib.dump(tfidf, config.ML_TFIDF_EXPORT_LOCATION)


def preprocess_labels_blended(data):
    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(data.classdescription).reshape(-1, 1)
    labels = ohe.fit_transform(labels).toarray()
    return labels, le, ohe


def preprocess_labels(labels):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    return labels


def batch_generator(X_train, y_train, batch_size):
    number_of_batches = X_train.shape[0] / batch_size
    counter = 0
    shuffle_index = np.arange(np.shape(y_train)[0])
    np.random.shuffle(shuffle_index)
    X =  X_train[shuffle_index, :]
    y =  y_train[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(np.array(X_batch), y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0


def get_embeddings(vocab):
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors
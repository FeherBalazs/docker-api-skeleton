from sklearn.externals import joblib
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model
import config

def load_ml_model():
    clf = joblib.load(config.CLASSIFIER_EXPORT_LOCATION)
    ml_tfidf = joblib.load(config.ML_TFIDF_EXPORT_LOCATION)
    return clf, ml_tfidf

def load_dl_model():
    clf = load_model(config.MODEL_EXPORT_LOCATION)
    dl_tfidf = joblib.load(config.DL_TFIDF_EXPORT_LOCATION)
    tokenizer = joblib.load(config.TOKENIZER_EXPORT_LOCATION)
    tokenizer.oov_token = None
    le = joblib.load(config.LABLE_ENCODER_EXPORT_LOCATION)
    return clf, dl_tfidf, tokenizer, le
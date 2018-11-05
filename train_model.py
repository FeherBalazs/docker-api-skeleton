import config
import numpy as np
import pandas as pd
import time
from time import time
import build_models
import preprocess_data
import plot_results
import dl_utils
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score


def train_classifier(model, X_train, y_train, sample_weight=None):
    print("Training {}...".format(model.__class__.__name__))
    start = time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    end = time()
    print("Done!\nTraining time (secs): {:.3f}".format(end - start))


def predict_labels(model, features, target):
    print("Predicting labels using {}...".format(model.__class__.__name__))
    start = time()
    y_pred = model.predict(features)
    end = time()
    print("Done!\nPrediction time (secs): {:.3f}".format(end - start))
    return [f1_score(target, y_pred, average='weighted'), accuracy_score(target, y_pred)]


def train_predict(model, X_train, y_train, X_test, y_test):
    print("------------------------------------------")
    print("Training set size: {}".format(X_train.shape[0]))
    train_classifier(model, X_train, y_train)
    training_set_results = predict_labels(model, X_train, y_train)
    test_set_results = predict_labels(model, X_test, y_test)
    print("F1 score for training set: {}\nAccuracy score for training set: {}".format(training_set_results[0],
                                                                                      training_set_results[1]))
    print("F1 score for test set: {}\nAccuracy score for test set: {}".format(test_set_results[0],
                                                                              test_set_results[1]))


def train_predict_dl_model(model, X_train, y_train, X_test, y_test, batch_size):
    batch_size = batch_size
    number_of_batches = X_train.shape[0] / batch_size
    model.fit_generator(dl_utils.batch_generator(X_train, y_train, batch_size), steps_per_epoch=number_of_batches, epochs=1)
    model.evaluate_generator(dl_utils.batch_generator(X_test, y_test, batch_size), steps=number_of_batches)


def train_dl_blended_model(model, padded_data, tfidf_features, labels, validation_split, epochs):
    early_stopping = EarlyStopping(monitor='val_acc',
                                   min_delta=0.005,
                                   patience=2,
                                   verbose=0,
                                   mode='auto')

    checkpointer = ModelCheckpoint(filepath=config.CHECKPOINTER_MODEL_EXPORT_LOCATION,
                                   verbose=1,
                                   save_best_only=True,
                                   monitor='val_acc')

    model.fit([padded_data, padded_data, padded_data, tfidf_features],
              labels,
              validation_split=validation_split,
              epochs=epochs,
              batch_size=config.BATCH_SIZE,
              callbacks=[checkpointer, early_stopping])

    model.load_weights(config.CHECKPOINTER_MODEL_EXPORT_LOCATION)

    return model


def predict_dl_blended_model(model, X_test_padded, X_test_tfidf, y_test, le):
    # Create predictions and save to DataFrame
    y_preds = model.predict([X_test_padded, X_test_padded, X_test_padded, X_test_tfidf])
    result_df = pd.DataFrame()
    result_df['y_preds'] = le.inverse_transform(np.argmax(y_preds, axis=1))
    result_df['y_test'] = le.inverse_transform(np.argmax(y_test, axis=1))

    print('F1 Score: ' + str(f1_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='weighted')))
    print('Overall Accuracy: ' + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1))))


def train_dl_model(data):
    print('Testing DL models...')
    X_train, X_test, y_train, y_test, padded_data, tfidf_features, labels, le, ohe, tfidf, X_train_padded, X_test_padded, X_train_tfidf, X_test_tfidf, tokenizer, train_classes = preprocess_data.preprocess_dl_data(data)

    # model = build_models.blended(config.MAXLEN,
    #                              config.VOCABULARY_SIZE,
    #                              X_train_tfidf,
    #                              labels,
    #                              embedding_dim=config.EMBEDDING_DIM,
    #                              dropout=config.DROPOUT)
    #
    # model_partial = train_dl_blended_model(model,
    #                                        X_train_padded,
    #                                        X_train_tfidf,
    #                                        y_train,
    #                                        validation_split=0.1,
    #                                        epochs=config.EPOCHS)
    #
    # predict_dl_blended_model(model_partial, X_test_padded, X_test_tfidf, y_test, le)

    print('Train on full dataset...')

    model = build_models.blended(config.MAXLEN,
                                 config.VOCABULARY_SIZE,
                                 X_train_tfidf,
                                 labels,
                                 embedding_dim=config.EMBEDDING_DIM,
                                 dropout=config.DROPOUT)

    model = train_dl_blended_model(model,
                                   padded_data,
                                   tfidf_features.todense(),
                                   labels,
                                   validation_split=0.1,
                                   epochs=config.EPOCHS)

    print('Test data on validation set...')
    validate_data(model, tfidf, train_classes, model_type='DL', tokenizer=tokenizer, le=le, ohe=ohe)
    print('Exporting models...')
    dl_utils.export_dl_model(model, tfidf, le)
    print('Setting up model for serving...')
    model._make_predict_function()
    print('Finished!')
    return model, tfidf, tokenizer, le


def validate_data(model, tfidf, train_classes, model_type, tokenizer=None, le=None, ohe=None):
    validate = open(config.VALIDATION_DATA_LOCATION, encoding='latin-1').read()
    validate = pd.read_json(validate, encoding='latin-1')
    validate = validate[validate.classdescription != 'Event management']
    validate = validate[validate.classdescription.isin(train_classes)]
    validate.rename(columns={'text': 'txt', 'ticketid': '_id'}, inplace=True)
    validate.dropna(inplace=True)

    result_df = pd.DataFrame()
    tfidf_features = tfidf.transform(validate.txt).toarray()

    if model_type == 'DL':
        sequences = tokenizer.texts_to_sequences(validate['txt'])
        padded_data_validate = pad_sequences(sequences, maxlen=config.MAXLEN)
        labels_validate = le.transform(validate.classdescription).reshape(-1, 1)
        labels_validate = ohe.transform(labels_validate).toarray()
        y_preds = model.predict([padded_data_validate, padded_data_validate, padded_data_validate, tfidf_features])
        result_df['y_preds'] = le.inverse_transform(np.argmax(y_preds, axis=1))
        result_df['y_test'] = le.inverse_transform(np.argmax(labels_validate, axis=1))
        print('F1 Score: ' + str(
            f1_score(np.argmax(labels_validate, axis=1), np.argmax(y_preds, axis=1), average='weighted')))
        print(
            'Overall Accuracy: ' + str(accuracy_score(np.argmax(labels_validate, axis=1), np.argmax(y_preds, axis=1))))

    elif model_type == 'ML':
        labels = validate.classdescription
        predicted = model.predict(tfidf_features)
        print('F1 Score: ' + str(f1_score(labels, predicted, average='weighted')))
        print('Overall Accuracy: ' + str(accuracy_score(labels, predicted)))


def train_ml_model(data):
    print('Testing ML models...')
    X_train, X_test, y_train, y_test, features, labels, tfidf, train_classes = preprocess_data.preprocess_ml_data(data)
    # model = build_models.build_logistic_regression_model()
    # train_predict(model, X_train, y_train, X_test, y_test)

    # print('Plotting results...')
    # model.fit(X_train, y_train)
    # plot_results(model, X_test, y_test)

    print('Fitting ML model on the whole dataset...')
    model = build_models.build_tuned_logistic_regression_model()
    model.fit(features, labels)
    print('Test data on validation set...')
    validate_data(model, tfidf, train_classes, model_type='ML')
    print('Exporting models...')
    dl_utils.export_ml_model(model, tfidf)
    print('Finished!')
    return model, tfidf
from flask import Flask, request, jsonify, render_template
import load_models
import dl_utils
import numpy as np
import train_model
import os

os.environ['KERAS_BACKEND'] = 'theano'

app = Flask(__name__)
model = None

# OUTPUT_FOLDER = os.path.join('static', 'model_output')
#
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#
# @app.route('/')
# @app.route('/results')
# def show_index():
#     full_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'cnf_matrix.png')
#     return render_template("results.html", user_image=full_filename)
#

@app.route("/predict_ml", methods=['POST'])
def predict_ml():
    results = {"success": False}
    if request.method == 'POST':
        try:
            data = request.get_json()
        except ValueError:
            return jsonify("Please pass some content!")

        results["predictions"] = []

        text = ml_tfidf.transform(data["text"]).toarray()
        predicted_class = clf.predict(text).tolist()
        predicted_probability = np.max(clf.predict_proba(text), axis=1).tolist()

        for (label, prob) in zip(predicted_class, predicted_probability):
            r = {"label": label, "probability": float(prob)}
            results["predictions"].append(r)

            results["success"] = True

    return jsonify(results)


@app.route("/retrain_ml", methods=['POST'])
def retrain_ml():
    results = {"success": False}
    if request.method == 'POST':
        try:
            data = request.get_json()
        except ValueError:
            return jsonify("Please pass some content!")

        global clf, ml_tfidf
        clf, ml_tfidf = train_model.train_ml_model(data)

        results["status"] = 'Model is retrained and is now available for predictor!'
        results["success"] = True

    return jsonify(results)


@app.route("/predict_dl", methods=['POST'])
def predict_dl():
    results = {"success": False}
    if request.method == 'POST':
        try:
            data = request.get_json()
        except ValueError:
            return jsonify("Please pass some content!")

        results["predictions"] = []

        tfidf_features, padded_data = dl_utils.transform_data(data, dl_tfidf, tokenizer)
        predicted_class = model.predict([padded_data, padded_data, padded_data, tfidf_features])
        predicted_class = le.inverse_transform(np.argmax(predicted_class, axis=1))
        predicted_probability = np.max(model.predict([padded_data, padded_data, padded_data, tfidf_features]), axis=1).tolist()

        for (label, prob) in zip(predicted_class, predicted_probability):
            r = {"label": label, "probability": float(prob)}
            results["predictions"].append(r)

            results["success"] = True

    return jsonify(results)


@app.route("/retrain_dl", methods=['POST'])
def retrain_dl():
    results = {"success": False}
    if request.method == 'POST':
        try:
            data = request.get_json()
        except ValueError:
            return jsonify("Please pass some content!")

        global model, dl_tfidf, tokenizer, le
        model, dl_tfidf, tokenizer, le = train_model.train_dl_model(data)
        model._make_predict_function()

        results["status"] = 'Model is retrained and is now available for predictor!'
        results["success"] = True

    return jsonify(results)
#
#
# @app.after_request
# def add_header(response):
#     response.cache_control.no_store = True
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '-1'
#     return response


model, dl_tfidf, tokenizer, le = load_models.load_dl_model()
model._make_predict_function()

clf, ml_tfidf = load_models.load_ml_model()

if __name__ == '__main__':
    # From Local
    app.run(debug=True, port=8080)

    # From Docker
    # app.run(host="0.0.0.0", debug=True, port=80)

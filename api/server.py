from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import pandas as pd
import dill as pickle
from training_v1 import PreProcessor, BeamSearch
from pandas import option_context

app = Flask(__name__)
cors = CORS(app)
app.config.from_file("settings.json", load=json.load)

@app.route("/predict", methods = ["POST"])
@cross_origin() 
def predict():
    if "Auth" not in request.headers:
         return unauthorized()
    if request.headers["Auth"] != app.config["API_KEY"]:
        return unauthorized()

    try:
        json_data = json.loads(request.data)
        data = {
            "HeartRate": [json_data["heartRate"]],
            "SkinConductance": [json_data["skinConductance"]],
            "EEG": [json_data["eeg"]],
            "Temperature": [json_data["temperature"]],
            "PupilDiameter": [json_data["pupilDiameter"]],
            "SmileIntensity": [json_data["smileIntensity"]],
            "FrownIntensity": [json_data["frownIntensity"]],
            "CortisolLevel": [json_data["cortisolLevel"]],
            "ActivityLevel": [json_data["activityLevel"]],
            "AmbientNoiseLevel": [json_data["ambientNoiseLevel"]],
            "LightingLevel": [json_data["lightingLevel"]]
        }
        req = pd.DataFrame(data)
    except Exception as e:
        return bad_request()
    
    classifier_eng = "model_eng_v1.sm"
    classifier_emo = "model_emo_v1.sm"

    print("Loading model.")
    model_eng = None
    model_emo = None
    with open("./models/" + classifier_eng, "rb") as file:
            model_eng = pickle.load(file)
    with open("./models/" + classifier_emo, "rb") as file:
            model_emo = pickle.load(file)
    print("Model loaded.")
    print("Making prediction.")
    prediction_eng = model_eng.predict(req)
    prediction_emo = model_emo.predict(req)

    print("Sending prediction.")
    response = jsonify(
        engagementLevel = prediction_eng[0],
        emotionalState = prediction_emo[0]
    )
    response.status_code = 200
    return response

@app.errorhandler(400)
def bad_request(error = None):
    message = {
        "status": 400,
        "message": "Bad request: " + request.url + " --> Something wrong with data."
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp

@app.errorhandler(401)
def unauthorized(error = None):
    message = {
        "status": 401,
        "message": "Unauthorized: " + request.url + " --> Wrong API key."
    }
    resp = jsonify(message)
    resp.status_code = 401
    return resp
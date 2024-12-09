import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import dill as pickle
import time

parallel = False
jobs = 4

data_path = "./data/emotional_monitoring_dataset_with_target.csv"
model_path_eng = "./models/model_eng_v1.sm"
model_path_emo = "./models/model_emo_v1.sm"

def train(target):
    dataset = pd.read_csv(data_path)
    dataset["EngagementLevel"] = dataset["EngagementLevel"].map({1: "Disengaged", 2: "Moderately Engaged", 3: "Highly Engaged"})

    X = dataset.drop(["EmotionalState", "CognitiveState", "EngagementLevel"], axis = 1)
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 777)

    pipe = make_pipeline(
        PreProcessor(),
        BeamSearch(),
        GradientBoostingClassifier()
    )

    boosting_params = {
        "gradientboostingclassifier__n_estimators": [5, 6, 7, 8, 9, 10],
    }

    grid = GridSearchCV(pipe, param_grid = boosting_params, scoring = "f1_macro", n_jobs = jobs if parallel else None)
    grid.fit(X_train, y_train)

    f1 = f1_score(y_test, grid.predict(X_test), average = "macro")

    return grid, f1

class PreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None, **fit_params):
        self.mean_heart_rate = X["HeartRate"].mean()
        self.mean_skin_conductance = X["SkinConductance"].mean()
        self.mean_eeg = X["EEG"].mean()
        self.mean_temperature = X["Temperature"].mean()
        self.mean_pupil_diameter = X["PupilDiameter"].mean()
        self.mean_smile_intensity = X["SmileIntensity"].mean()
        self.mean_frown_intensity = X["FrownIntensity"].mean()
        self.mean_cortisol_level = X["CortisolLevel"].mean()
        self.mean_activity_level = X["ActivityLevel"].mean()
        self.mean_noise_level = X["AmbientNoiseLevel"].mean()
        self.mean_light_level = X["LightingLevel"].mean()
        return self

    def transform(self, X):
        X["HeartRate"] = X["HeartRate"].fillna(self.mean_heart_rate)
        X["SkinConductance"] = X["SkinConductance"].fillna(self.mean_skin_conductance)
        X["EEG"] = X["EEG"].fillna(self.mean_eeg)
        X["Temperature"] = X["Temperature"].fillna(self.mean_temperature)
        X["PupilDiameter"] = X["PupilDiameter"].fillna(self.mean_pupil_diameter)
        X["SmileIntensity"] = X["SmileIntensity"].fillna(self.mean_smile_intensity)
        X["FrownIntensity"] = X["FrownIntensity"].fillna(self.mean_frown_intensity)
        X["CortisolLevel"] = X["CortisolLevel"].fillna(self.mean_cortisol_level)
        X["ActivityLevel"] = X["ActivityLevel"].fillna(self.mean_activity_level)
        X["AmbientNoiseLevel"] = X["AmbientNoiseLevel"].fillna(self.mean_noise_level)
        X["LightingLevel"] = X["LightingLevel"].fillna(self.mean_light_level)
        return X
    
class BeamSearch(BaseEstimator, TransformerMixin):
    def __init__(self, beam = 5, folds = 5, gb_estimators = 5, stop = 2):
        self.beam = beam
        self.folds = folds
        self.gb_estimators = gb_estimators
        self.stop = stop

    def fit(self, X, y):
        n_features = len(X.columns)
        R = [[i] for i in range(n_features)]
        best_score = 0
        best_features = []
        best_dim = 0
        for j in range(n_features):
            cv_score = [self.evaluate_score(X, y, J) for J in R]
            sorted_cv_score, sorted_R = zip(*[(b, a) for b, a in sorted(zip(cv_score, R), reverse = True)])
            R = list(sorted_R[0:self.beam])
            if sorted_cv_score[0] > best_score:
                best_score = sorted_cv_score[0]
                best_features = R[0]
                best_dim = j
            if j - best_dim >= self.stop:
                self.best_score = best_score
                self.best_features = best_features
                return self
            for i in range(len(R)):
                J = R.pop(0)
                for f in list(set(range(n_features)) - set(J)):
                    R.append([*J, f])
        self.best_score = best_score
        self.best_features = best_features
        return self

    def evaluate_score(self, X, y, J):
        cols = X.columns.values.tolist()
        subcols = [cols[i] for i in J]
        X_sub = X[subcols]
        gb = GradientBoostingClassifier(n_estimators = self.gb_estimators)
        return sum(cross_val_score(gb, X_sub, y, cv = self.folds, scoring = "f1_macro"))

    def transform(self, X):
        cols = X.columns.values.tolist()
        subcols = [cols[i] for i in self.best_features]
        X_sub = X[subcols]
        return X_sub
    
if __name__ == "__main__":
    print("Started training for estimating Engagement Level.")
    t1 = time.time()
    grid_eng, f1_eng = train("EngagementLevel")
    t2 = time.time()
    seconds = (t2 - t1) // 1
    print(f"Training completed in {int(seconds // 60)} minutes {int(seconds % 60)} seconds.")

    print(f"Best parameters: {grid_eng.best_params_}")

    print(f"F1 score: {f1_eng}")

    print(f"Selected features: {grid_eng.best_estimator_["beamsearch"].best_features}")

    print("Started training for estimating Emotional State.")
    t1 = time.time()
    grid_emo, f1_emo = train("EmotionalState")
    t2 = time.time()
    seconds = (t2 - t1) // 1
    print(f"Training completed in {int(seconds // 60)} minutes {int(seconds % 60)} seconds.")

    print(f"Best parameters: {grid_emo.best_params_}")

    print(f"F1 score: {f1_emo}")

    print(f"Selected features: {grid_emo.best_estimator_["beamsearch"].best_features}")

    model_eng = grid_eng.best_estimator_
    model_emo = grid_emo.best_estimator_

    print("Serializing models.")
    with open(model_path_eng, "wb") as file:
        pickle.dump(model_eng, file)
    with open(model_path_emo, "wb") as file:
        pickle.dump(model_emo, file)
    print("Serializing complete.")
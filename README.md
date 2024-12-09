# Engagement level and emotional state classifier

## Dataset description

This classification model(s) was(were) based on this [kaggle](https://www.kaggle.com/datasets/ziya07/emotional-monitoring-dataset) dataset.

Here is a description of this dataset copied from kaggle.

> This dataset is designed for emotional monitoring and feedback systems, specifically tailored for university ideological and political education using biosensor technology. It aims to simulate the physiological and behavioral responses of students to track their engagement levels in educational environments.
>
> Features:
> HeartRate: The student's heart rate, measured in beats per minute (bpm). This is indicative of the student's emotional and physical state, with lower rates typically associated with calm and engaged states, and higher rates with stress or disengagement.
> 
> SkinConductance: Skin conductance (also known as galvanic skin response or GSR), measured in microsiemens (µS). This reflects emotional arousal, with higher values correlating to higher emotional engagement or stress.
> 
> EEG (Electroencephalography): The frequency of brain waves recorded through EEG. Higher EEG values indicate focused mental engagement, while lower values can suggest distraction or disengagement.
>
> Temperature: The body temperature of the student, measured in degrees Celsius. Temperature changes can indicate stress or disengagement, with higher values potentially pointing to physical stress or tension.
> 
> PupilDiameter: The diameter of the student’s pupil, measured in millimeters (mm). Pupil dilation is a common indicator of cognitive load and emotional engagement, with larger pupils generally indicating higher focus or emotional response.
> 
> SmileIntensity: The intensity of the student’s smile, quantified on a scale from 0.0 (no smile) to 1.0 (maximum smile). This feature reflects positive emotional responses and engagement levels.
> 
> FrownIntensity: The intensity of the student’s frown, quantified on a scale from 0.0 (no frown) to 1.0 (maximum frown). A higher frown intensity typically correlates with disengagement or negative emotions.
> 
> CortisolLevel: The concentration of cortisol, a stress hormone, measured in micrograms per deciliter (µg/dL). High cortisol levels are associated with stress or disengagement, while lower levels indicate a relaxed or engaged state.
> 
> ActivityLevel: The level of physical activity, measured in steps or movement intensity. This reflects how physically active the student is, which can correlate with emotional engagement or stress.
> 
> AmbientNoiseLevel: The background noise level in decibels (dB). High noise levels can affect a student's ability to focus, which may impact emotional and cognitive engagement.
> 
> LightingLevel: The lighting intensity in lux. This feature can influence engagement, as poorly lit environments might lead to lower attention levels.
> 
> Target Column:
>
> EngagementLevel: A categorical target column that indicates the level of student engagement during the educational session:
> 
> 1: Highly Engaged – This level is assigned when the student shows low heart rate, high skin conductance, high EEG levels, and other indicators of deep emotional and cognitive engagement.
> 
> 2: Moderately Engaged – Assigned when the student demonstrates moderate physiological responses, indicating partial focus or attention.
> 
> 3: Disengaged – Assigned when the student displays signs of stress or low engagement, with higher heart rate, low skin conductance, and low EEG values.

## Model

Exploratory data analysis and modeling is introduced in notebook [`notebooks/EDA+Model.ipynb`](notebooks/EDA+Model.ipynb)

We can say that GradientBoosting classifier solving this problem better than other classifiers. So I stopped on this method.

In the [`notebooks/Pipeline.ipynb`](notebooks/Pipeline.ipynb) described machine learning pipeline.

It consists of:
- PreProcessor. It needs to preprocess data: to eliminate NA with mean values.
- BeamSearch. Feature selection with beam search algorithm.
- GradientBoostingClassifier. Classification model.

Whole pipeline and training routine were carried in [`api/training_v1.py`](api/training_v1.py) file.

Inside it has variables to change data path and output path.

Also it has bool variable `parallel`. If `True` it enables multiprocessing training with `jobs` number of jobs. By default it set up to `False`.

To start training simply open console in `api/` directory and type in `python training_v1.py`. The output models files will be in `api/models/` directory.

## API

Also there is basic API constructed with Flask.

It has only one endpoint: `/predict`.

The input of this endpoint is json with all of the features of dataset.

It outputs two predictions: Engagement level and emotional state.

Also this server checks `API_KEY` to authorize request. `API_KEY` can be configured in [`api/settings.json`](api/settings.json) file.

You can see work of this API by checking [`notebooks/RequestTest.ipynb`](notebooks/RequestTest.ipynb) notebook.

You can start the API by opening `start.bat` file. It will use `waitress` server to host api. It will host on `http://127.0.0.1:7070`

## Site

Also there is html/css/js site that have a simple form to send request to an API.

## Requirements

You can see requirements of this project in [`requirements.txt`](requirements.txt)

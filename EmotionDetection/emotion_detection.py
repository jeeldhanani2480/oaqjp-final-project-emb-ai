"""Emotion detection using the Watson NLP EmotionPredict function."""
import json
import requests


def emotion_detector(text_to_analyze):
    """Analyze the emotions in the given text using Watson NLP.

    Returns a dictionary with the scores for anger, disgust, fear, joy and
    sadness, plus the name of the dominant emotion. If the input is blank
    (the service returns status code 400), every value is None.
    """
    url = ('https://sn-watson-emotion.labs.skills.network/v1/'
           'watson.runtime.nlp.v1/NlpService/EmotionPredict')
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, json=myobj, headers=headers, timeout=30)

    # Blank entries: the service responds with status code 400
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Convert the response text into a dictionary
    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']

    # Extract the required set of emotions and their scores
    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']

    emotion_scores = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }

    # The dominant emotion is the emotion with the highest score
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }

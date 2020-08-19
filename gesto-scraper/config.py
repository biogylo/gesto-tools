
PRE_DATABASE_LOCATION = "statistics/database.csv"
PRE_IMAGES_LOCATION = "images/"
PRE_STATISTICS_LOCATION = "statistics/"
PRE_JSONS_LOCATION = "jsons/"
PRE_IMAGES_REGEX = "(.+\d\d\d\d)(happyb|ängry|angry|angryb|surprised|surprisedb|sadb|neutralb|sad|neutral|happy|test)(\d+)\.webp"
DATASET_LOCATION = "gesto-dataset/"

import os
for location in [PRE_STATISTICS_LOCATION,DATASET_LOCATION]:
    if not os.path.exists(location):
        os.makedirs(location)

FIX_EMOTIONS = {
"happy":"happy",
"happyb":"happy",

"sad":"sad",
"sadb":"sad",

"surprised":"test",
"surprisedb":"test",

"neutral":"neutral",
"neutralb":"neutral",

"angry":"angry",
"angryb":"angry",
"ängry":"angry",
"ängryb":"angry",
"test":"test",
"testb":"test"
}

EMOTIONS = ["neutral","happy","sad","test","angry"]

EMOTION_COLUMNS = [
                    "valid_neutral","valid_happy","valid_sad","valid_test","valid_angry",
                    "invalid_neutral","invalid_happy","invalid_sad","invalid_test","invalid_angry",
                    "valid_total","invalid_total"
                   ]

###FACE TRANSFORMER
FACE_SHAPE_PREDICTOR_LOCATION = 'dlib/shape_predictor_68_face_landmarks.dat'
FACE_DETECTOR_LOCATION = 'dlib/mmod_human_face_detector.dat'

print("""Gesto Dataset preprocessing Script
\tThis script will do all the necessary preprocessing steps, to construct
the Gesto Dataset""")

"""

Steps needed in the preprocessing
    1. Load the CSV file and remove repeated entries, the surviving entry will be
    that with a corresponding group of pictures in the pictures

    2. Remove background, apply affine transformation, and histogram normalization

    3. Create a new dataset with all the files on different folders, in this way
    gesto-dataset/participant0/angry/0.webp
                              /angry/10.webp
                              /happy/4.webp
                /participant23/sad/5.webp
                              /surprised/7.webp

                /statistics/average_face.webp
                           /etc
"""

print("""\nStep 0: Imports""")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import re
import os
import cv2
from tqdm import tqdm
import face_transformer as ftf

pre_database = pd.read_csv(cfg.PRE_DATABASE_LOCATION)
pre_database["complete"] = False

for filename in tqdm(os.listdir(cfg.PRE_IMAGES_LOCATION)):
    try:
        cuserid = re.match(cfg.PRE_IMAGES_REGEX,filename).group(1)
        pre_database.loc[pre_database.userid == cuserid,"complete"] =  True
    except:
        print("\tError in "+filename)

print("Imported successfully")

print("""\nStep 1: Removing repeated and nonparticipating entries""")

pre_database.to_csv("gesto-information/pre_database.csv")
post_database = pre_database[pre_database.complete == True]
post_database.to_csv("gesto-information/post_database.csv")

non_database = pre_database[pre_database.complete == False]
non_database.to_csv("gesto-information/nonparticipating.csv")

failed_database = non_database[~non_database.email.isin(post_database.email)]
failed_database.to_csv("gesto-information/failed.csv")

repeated_database = non_database[non_database.email.isin(post_database.email)]
repeated_database.to_csv("gesto-information/repeated.csv")

print(f"""\tSummary:
\t\t{len(pre_database.index)} entries loaded
\t\t{len(non_database.index)} entries removed
\t\t\t{len(failed_database.index)} did not do the experiment (failed)
\t\t\t{len(repeated_database.index)} json file was repeated (repeated)
\t\t{len(post_database.index)} valid entries available in the dataset""")

shuffled_database = post_database.sample(frac=1,random_state=12344321).reset_index(drop=True)
shuffled_database.to_csv("gesto-information/shuffled.csv")

print(shuffled_database.head())

print("""\n\nStep 2: Applying image processing and sorting in folders""")

bad_images = pd.DataFrame(columns=["userid","filename","reason","emotion"])
good_images = pd.DataFrame(columns=["userid","filename","emotion"])

for emotion in cfg.EMOTIONS:
    shuffled_database["valid_"+emotion] = 0
    shuffled_database["invalid_"+emotion] = 0
    shuffled_database["valid_total"] = 0
    shuffled_database["invalid_total"] = 0
    shuffled_database["total"] = 0
    shuffled_database["clarity"] = 0

try:
    os.makedirs("gesto-dataset/bad-images/")
except:
    pass

for filename in tqdm(os.listdir(cfg.PRE_IMAGES_LOCATION)):

    userid = re.match(cfg.PRE_IMAGES_REGEX,filename).group(1)
    emotion = cfg.FIX_EMOTIONS[re.match(cfg.PRE_IMAGES_REGEX,filename).group(2)]

    pic_id = re.match(cfg.PRE_IMAGES_REGEX,filename).group(3)
    try:
        id = shuffled_database[shuffled_database.userid == userid].index[0]
    except:
        print("Not found error in "+filename)
        continue
    img = cv2.imdecode(np.fromfile(cfg.PRE_IMAGES_LOCATION + filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    blur = ftf.get_blurryness(img)

    if shuffled_database.loc[id,"stretchx"] > 0:
        sx = np.float(shuffled_database.loc[id,"stretchx"])
        sy = np.float(shuffled_database.loc[id,"stretchy"])
        dsize = (int(640*sx),int(480*sy))
        img = cv2.resize(img,dsize)
    fixed = ftf.get_fixed_img(img)
    shuffled_database.loc[id,"total"] += 1

    if fixed is not None:
        shuffled_database.loc[id,"clarity"] += blur

        filepath = f"gesto-dataset/subject{str(id)}/{emotion}/"

        try:
            os.makedirs(filepath)
        except:
            pass
        filepath += f"{str(shuffled_database.loc[id,'valid_' + emotion])}.png"

        cv2.imwrite(filepath,fixed)

        shuffled_database.loc[id,"valid_" + emotion] += 1
        shuffled_database.loc[id,"valid_total"] += 1

    else:

        filepath = f"gesto-dataset/bad-images/subject{str(id)}{emotion}{str(shuffled_database.loc[id,'invalid_' + emotion])}"
        cv2.imwrite(filepath +".png",img)

        shuffled_database.loc[id,"invalid_" + emotion] += 1
        shuffled_database.loc[id,"invalid_total"] += 1

    shuffled_database.to_csv("gesto-information/shuffled_complete.csv")

print("Dataset built")

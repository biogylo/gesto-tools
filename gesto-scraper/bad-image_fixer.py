import re
import config as cfg
bad_image_regex = r"(subject\d+)(angry|sad|test|happy|neutral)(\d+\.png)"

print("Reevaluating bad images")
good = 0
bad = 0
import cv2
import face_transformer as ftf
from tqdm import tqdm
import os
import numpy as np

for filename in tqdm(os.listdir("gesto-dataset/bad-images/")):
    fields = re.match(bad_image_regex, filename).groups([1,2,3])
    img = cv2.imdecode(np.fromfile("gesto-dataset/bad-images/" + filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    fixed = ftf.get_fixed_img(img)
    filepath =f"gesto-dataset/{fields[0]}/{fields[1]}/{fields[2]}"
    if fixed is not None:
        good+=1
        cv2.imshow("",fixed)
        cv2.waitKey(100)
        cv2.imwrite(filepath,fixed)
    else:
        bad+=1
print(f"Pictures fixed = {good}/{good+bad}, {100*good/(good+bad)}%")

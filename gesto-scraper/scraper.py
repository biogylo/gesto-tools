import config as cfg
import os
import json
from tqdm import tqdm
print("Parsing jsons")
content = []
for file in tqdm(os.listdir(cfg.PRE_JSONS_LOCATION)):
    with open(cfg.PRE_JSONS_LOCATION+file, encoding="utf8") as datapoint:
        print(file)
        content.append(json.load(datapoint))

print("Saving")
import pandas as pd
database = pd.DataFrame(content)
database.to_csv(cfg.PRE_DATABASE_LOCATION)

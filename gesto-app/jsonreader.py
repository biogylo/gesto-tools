import json
import os
import pandas as pd

jsonpath = os. getcwd() + '/jsons/'

jsonlist = []
for jsonfilename in os.listdir(jsonpath):
    jsonfile = open(jsonpath+jsonfilename)
    jsonlist.append(json.load(jsonfile))
    jsonfile.close()
df = pd.DataFrame(jsonlist)

df.to_csv("all_csvs.csv",  index=False,encoding='utf-8')

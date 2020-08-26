#!/usr/bin/python3

import json
from sys import argv

inputArgs = argv[1:]    # Terminal command = './separate_json.py *.json'

for arg in inputArgs:
    with open(arg) as f:
        data = json.load(f)

        SUBJ_ID = list(data.keys())

        for subj in SUBJ_ID:
            with open('.'.join([subj, 'json']), 'w') as fw:
                json.dump({subj:data[subj]}, fw, indent=4)



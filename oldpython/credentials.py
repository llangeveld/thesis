import json

credentials = {}
credentials['CONSUMER_KEY'] = "lXhqyEZRfvpM4GzM6Vuc7ag9p"
credentials['CONSUMER_SECRET'] = "gdKFu1Yu3MFZwy9Y9mLnouo77aulE0w6VCKARr2oT3HkyyHsp3"
credentials['ACCESS_TOKEN'] = "441148345-f2RQbRGtlKr4xD9QjbMWvld2pHmtCgBEVIgs1VDI"
credentials['ACCESS_SECRET'] = "rCqlZlcoxE5UaV2ReYhxkAOdUbEdkf7ewuxnm6dR63jiH"

with open("credentials.json", "w") as f:
    json.dump(credentials, f)
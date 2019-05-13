#!/usr/bin/python3

import tweepy
import json
import csv

with open("credentials.json", "r") as f:
    creds = json.load(f)

auth = tweepy.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
api = tweepy.API(auth, wait_on_rate_limit=True)

search = tweepy.Cursor(api.search, q="#MeToo women -filter:replies -filter:retweets", tweet_mode="extended", lang="en").items(3000)

with open("tweets2.csv", "a") as csvFile:
    writer = csv.writer(csvFile)
    count = 0
    for item in search:
        writer.writerow([item.id, item.author._json['name'], item.full_text])
        count += 1

print(count)


#!/usr/bin/python3
from sklearn.metrics import cohen_kappa_score
import csv

with open("resources/annotate300.csv") as f:
        hielke = [line[2] for line in csv.reader(f, delimiter=",")]

with open("resources/annotateLon300.csv") as f:
        lonneke = [line[2] for line in csv.reader(f, delimiter=",")]

print(cohen_kappa_score(hielke, lonneke))
i = 0
nietGelijknummers = []
for item in hielke:
    if item != lonneke[i]:
        nietGelijknummers.append(i+1)
    i += 1

print(nietGelijknummers)
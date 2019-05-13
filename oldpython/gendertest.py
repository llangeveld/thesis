#!/usr/bin/python3

import csv

with open("resources/names_annotated.csv") as f:
    genderFile = [line for line in csv.reader(f, delimiter=",")]

fCount = 0
mCount = 0
nCount = 0
femaleNames = []
maleNames = []
for name, gender in genderFile:
    if gender == "F":
        fCount += 1
        femaleNames.append(name)
    elif gender == "M":
        mCount += 1
        maleNames.append(name)
    elif gender == "N":
        nCount += 1

print("Female: {0} | Male: {1} | Neutral: {2}".format(fCount, mCount, nCount))

import csv

def get_gender():
    with open("resources/names_annotated.csv") as f:
        genderFile = [line for line in csv.reader(f, delimiter=",")]
    femaleNames = []
    maleNames = []
    for name, gender in genderFile:
        if gender == "F":
            femaleNames.append(name)
        elif gender == "M":
            maleNames.append(name)

    return(femaleNames, maleNames)

def main():
    femaleNames, maleNames = get_gender()
    

    
#!/usr/bin/python3
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
    print(femaleNames, maleNames)

#!/usr/bin/python3

import csv
import re

def main():
    with open("resources/feminism.csv") as f:
        feminismFile = [[line[0], line[2]] for line in csv.reader(f, delimiter=",")]

    with open("resources/feminismStance.csv", "a") as newFile:
        writer = csv.writer(newFile)
        for item in feminismFile:
            text = re.sub(r"#SemST", "", item[0])
            noHashtags = re.sub(r"#", "", text)
            if item[1] == "FAVOR":
                writer.writerow([noHashtags, "TBD"])

if __name__ == "__main__":
    main()
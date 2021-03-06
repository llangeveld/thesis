#!/usr/bin/python3
import csv

def get_files():
    with open("resources/test.csv", 'rt') as f1:
        testFile = [line for line in csv.reader(f1, delimiter=",")]
    with open("resources/train.csv", 'rt') as f2:
        trainFile = [line for line in csv.reader(f2, delimiter=",")]
    f = testFile + trainFile
    return f

def main():
    f = get_files()
    with open("feminism.csv", "w+") as csvFile:
        writer = csv.writer(csvFile)
        for el in f:
            if target == "Feminist Movement":
                writer.writerow([el[0], el[1], el[2]])

if __name__ == "__main__":
    main()
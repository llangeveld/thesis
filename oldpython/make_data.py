#!/usr/bin/python3
import csv
import random


def import_data():
    with open("resources/finalData.csv") as f:
        f1 = [line for line in csv.reader(f, delimiter=",")]

    return f1


def main():
    dataMT = import_data()
    random.shuffle(dataMT)
    cutOff = int(0.7*len(dataMT))
    train = dataMT[:cutOff]
    test = dataMT[cutOff:]

    with open("resources/train.csv", "w+") as f1:
        writer = csv.writer(f1)
        for line in train:
            writer.writerow(line)

    with open("resources/test.csv", "w+") as f2:
        writer = csv.writer(f2)
        for thing in test:
            writer.writerow(thing)

if __name__ == "__main__":
    main()
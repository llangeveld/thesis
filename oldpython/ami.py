#!/usr/bin/python3
import csv

def get_files():
    with open("resources/ami/en_testing_labeled.tsv", 'rt') as f1:
        testFile = [line for line in csv.reader(f1, delimiter="\t")]
    with open("resources/ami/en_training.tsv", 'rt') as f2:
        trainFile = [line for line in csv.reader(f2, delimiter="\t")]
    f = testFile + trainFile
    return f

def main():
    f = get_files()
    with open("resources/ami.csv", "w+") as csvFile:
        writer = csv.writer(csvFile)
        for el in f:
            if stance == "1":
                writer.writerow([el[1], "A"])
            else:
                writer.writerow([el[1], "TBD"])


if __name__ == "__main__":
    main()
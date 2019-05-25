#!/usr/bin/python3
import csv
import itertools

def get_gender():
    """
    :return: two lists, one of female names and one of male names
    Function reads out annotated names-file, and makes two lists
    (one with female, and one with male names).
    """
    with open("resources/names_annotated.csv") as f:
        genderFile = [line for line in csv.reader(f, delimiter=",")]
    femaleNames = []
    maleNames = []
    for name, gender in genderFile:
        if gender == "F":
            femaleNames.append(name)
        elif gender == "M":
            maleNames.append(name)
    return femaleNames, maleNames

def classify_gender(femaleNames, maleNames, tweetFile):
    """
    :param: femaleNames: A list of female names
    :param: maleNames: A list of male names
    :param: tweetFile: A file with the raw tweets [tweetID, author, text]
    :return: genderedFile: A file with the raw tweets, with gender
    included [tweetID, author, text, gender]
    """
    genderedFile = []
    tweetIDs = []
    for tweetID, tweetAuthor, tweetText in tweetFile:
        if tweetID not in tweetIDs:
            tweetAuthor = tweetAuthor.lower()
            namesList = []
            for item in itertools.zip_longest(femaleNames, maleNames):
                femaleName = item[0]
                maleName = item[1]
                if maleName is not None and maleName.lower() in tweetAuthor:
                    namesList.append(maleName)
                elif femaleName.lower() in tweetAuthor:
                    namesList.append(femaleName)
            if namesList != []:
                name = max(namesList, key=len)
                if name in femaleNames:
                    genderedFile.append([tweetID, tweetAuthor, tweetText, "F"])
                elif name in maleNames:
                    genderedFile.append([tweetID, tweetAuthor, tweetText, "M"])
            tweetIDs.append(tweetID)
    return genderedFile


def import_data():
    femaleNames, maleNames = get_gender()
    with open("resources/annotate1000.csv") as f:
        annotationFile = [line for line in csv.reader(f, delimiter=",")]  # [tweetID, tweetText, stance]
    with open("resources/tweets.csv") as f2:  # [tweetID, tweetAuthor, tweetText]
        tweetFile = [line for line in csv.reader(f2, delimiter=",")]
    genderedFile = classify_gender(femaleNames, maleNames, tweetFile) # [tweetID, tweetAuthor, tweetText, gender]
    completeFile = []
    IDS = []
    for line in annotationFile:
        for otherLine in genderedFile:
            if line[0] == otherLine[0] and line[0] not in IDS:
                completeFile.append([line[0], otherLine[2], otherLine[3], line[2]])
                IDS.append(line[0])
    return completeFile

def write_csv(f):
    with open("finalData.csv", "w+") as csvFile:
        writer = csv.writer(csvFile)
        for line in f:
            writer.writerow(line)

def main():
    f = import_data()
    write_csv(f)

if __name__ == "__main__":
    main()

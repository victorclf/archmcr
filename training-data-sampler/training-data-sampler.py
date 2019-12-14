#!/usr/bin/env python3

import os
import sys
import csv
import random

# START WORKAROUND https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
# END WORKAROUND https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072

DATASET_PATH = '../preprocess-dataset/output/reviews.csv'
TRAINING_DATASET_SIZE = 1000
OUTPUT_PATH = 'output/discussions-sample-%d.csv' % TRAINING_DATASET_SIZE


def getRandomSampleOfDiscussions(sampleSize):
	discussions = []
	with open(DATASET_PATH, 'r') as fin:
		cfin = csv.reader(x.replace('\0', '') for x in fin) #ignore NULL bytes, which crash the csv.reader
		next(cfin, None) # skip the first row, which contains the header
		for row in cfin:
			discussions.append(row)
	sampleDiscussions = random.choices(discussions, k = sampleSize)
	return sampleDiscussions

	
def writeDiscussionsToCSV(discussions, outputPath):
	with open(DATASET_PATH, 'r') as fin:
		cfin = csv.reader(x.replace('\0', '') for x in fin) #ignore NULL bytes, which crash the csv.reader
		header = next(cfin)
	
	with open(outputPath, 'w') as fout:
		cfout = csv.writer(fout)
		cfout.writerow(header)
		cfout.writerows(discussions)


if __name__ == '__main__':
	sampleDiscussions = getRandomSampleOfDiscussions(TRAINING_DATASET_SIZE)
	writeDiscussionsToCSV(sampleDiscussions, OUTPUT_PATH)

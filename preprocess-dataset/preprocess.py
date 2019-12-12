#!/usr/bin/env python3

import os
import csv
import sys
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


CSV_FIELDS = ['reviewRequestId', 'repository', 'reviewRequestSubmitter', 'reviewId', 'diffCommentId', 'replyId', 
		'replyDiffCommentId', 'type',  'username', 'timestamp', 'text']

DATASET_FOLDER = '../rbminer/downloads'
OUTPUT_PATH = 'output/reviews.csv'

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def countLines(csvPath):
	with open(csvPath, 'r') as fin:
		return len(fin.readlines()) - 1 #don't count the header
		

def countCSVRows(csvPath):
	with open(csvPath, 'r') as fin:
		#cfin = csv.reader(fin)
		cfin = csv.reader(x.replace('\0', '') for x in fin) #ignore NULL bytes, which crash the csv.reader
		return sum(1 for row in cfin) - 1 #don't count the header
						

# ~ def analyze():
	# ~ csvFiles = {}
	# ~ for root, dirs, files in os.walk(DATASET_FOLDER):
		# ~ for f in files:
			# ~ if f.startswith('review') and f.endswith('csv'):
				# ~ csvPath = os.path.join(root, f)
				# ~ eprint(csvPath)
				# ~ numRows = countCSVRows(csvPath)
				# ~ #eprint('%s %d' % (f, numRows))
				# ~ csvFiles[numRows] = csvFiles.get(numRows, 0) + 1
	# ~ return csvFiles
	

def mergeCSV(csvFilePaths, outputPath):
	for csvPath in csvFilePaths:
		if os.path.exists(csvPath):
			with open(csvPath, 'r') as fin:
				headerLine = fin.readline()
				
				if not os.path.exists(outputPath):
					with open(outputPath, 'w') as fout:
						fout.write(headerLine)
				
				with open(outputPath, 'a') as fout:
					for line in fin:
						fout.write(line)
						

def getReviewsWithMultipleComments():
	csvFiles = []
	for root, dirs, files in os.walk(DATASET_FOLDER):
		for f in files:
			if f.startswith('review') and f.endswith('csv'):
				csvPath = os.path.join(root, f)
				numComments = countCSVRows(csvPath)
				if numComments > 1:
					eprint(f)
					csvFiles.append(csvPath)
	return csvFiles


if __name__ == '__main__':
	# ~ rowCounts = analyze()
	# ~ for r in sorted(rowCounts):
		# ~ print(r, rowCounts[r])
	mergeCSV(getReviewsWithMultipleComments(), OUTPUT_PATH)

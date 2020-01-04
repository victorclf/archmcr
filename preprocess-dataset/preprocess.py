#!/usr/bin/env python3

'''
This script prepares the data for the binary classifier of design discussions. It mainly does two things:
(1) concatenates all reviews with more than one comment into a single .csv file (Easier to select a random sample for training and for passing as input to the automatic classifier later).
(2) merges DiffComment with their associated ReplyDiffComment (this is because it is hard to understand ReplyDiffComment separated from their original thread of discussion).
'''

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
						

def getReviewsWithMultipleComments():
	csvFiles = []
	for root, dirs, files in os.walk(DATASET_FOLDER):
		for f in files:
			if f.startswith('review') and f.endswith('csv'):
				csvPath = os.path.join(root, f)
				numComments = countCSVRows(csvPath)
				if numComments > 1:
					csvFiles.append(csvPath)
	return csvFiles


def readCSVRowsAsDict(csvPath):
	with open(csvPath, 'r') as fin:
		dcfin = csv.DictReader(x.replace('\0', '') for x in fin) #ignore NULL bytes, which crash the csv.reader
		return [row for row in dcfin]
			

''' 
Merge ReplyDiffComment nodes with parent DiffComment.
'''
def mergeDiffCommentsWithReplies(rows):
	mergedRows = []
	for row in rows:
		if mergedRows and row['type'] == 'ReplyDiffComment' and mergedRows[-1]['type'] == 'DiffComment':
			mergedRows[-1]['text'] += '\n\n' + row['text']
		else:
			mergedRows.append(row)
	return mergedRows
	
''' 
Merge Review.reply.body_top nodes with parent Review.body_top.
'''
def mergeReviewReplyBodyTop(rows):
	mergedRows = []
	for row in rows:
		if mergedRows and row['type'] == 'Review.reply.body_top' and mergedRows[-1]['type'] == 'Review.body_top':
			mergedRows[-1]['text'] += '\n\n' + row['text']
		else:
			mergedRows.append(row)
	return mergedRows


''' 
Merge Review.reply.body_bottom nodes with parent Review.body_bottom.
'''
def mergeReviewReplyBodyBottom(rows):
	mergedRows = []
	for row in rows:
		if mergedRows and row['type'] == 'Review.reply.body_bottom' and mergedRows[-1]['type'] == 'Review.reply.body_bottom':
			mergedRows[-1]['text'] += '\n\n' + row['text']
		elif mergedRows and row['type'] == 'Review.body_bottom' and mergedRows[-1]['type'] == 'Review.reply.body_bottom':
			row['text'] += '\n\n' + mergedRows[-1]['text']
			mergedRows.pop()
			mergedRows.append(row)
		else:
			mergedRows.append(row)
	return mergedRows


def createCombinedCSV(outputPath):
	with open(outputPath, 'w') as fout:
		fout.write(','.join(CSV_FIELDS))
		fout.write('\n')


def appendToCombinedCSV(outputPath, rows):
	with open(outputPath, 'a') as fout:
		dcfout = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
		for row in rows:
			dcfout.writerow(row)


if __name__ == '__main__':
	# ~ rowCounts = analyze()
	# ~ for r in sorted(rowCounts):
		# ~ print(r, rowCounts[r])
		
	if os.path.exists(OUTPUT_PATH):
		raise Exception("Error: Output file already exists.")
	createCombinedCSV(OUTPUT_PATH)
	eprint('Selecting reviews with multiple comments...')
	reviewPaths = getReviewsWithMultipleComments()
	#reviewPaths = [os.path.join(DATASET_FOLDER, 'review-11983.csv')]
	for path in reviewPaths:
		eprint('Processing', path)
		rows = readCSVRowsAsDict(path)
		rows = mergeDiffCommentsWithReplies(rows)
		rows = mergeReviewReplyBodyTop(rows)
		rows = mergeReviewReplyBodyBottom(rows)
		appendToCombinedCSV(OUTPUT_PATH, rows)

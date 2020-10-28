#!/usr/bin/env python3
import os
import sys
import subprocess
import csv

def getBasicAnalysisStats(basePath):
	totalReviewRequests = 0
	totalComments = 0
	totalWords = 0
	for root, dir, files in os.walk(basePath):
		for f in files:
			if f.startswith('review') and f.endswith('txt'):
				filePath = os.path.join(root, f)
				comments = subprocess.check_output(f"iconv -f utf-16 -t utf-8 < {filePath} | grep -E '\$\$\$' | wc -l", shell=True)
				words = subprocess.check_output(f"iconv -f utf-16 -t utf-8 < {filePath} | sed -e '/^\s*\$/d;/^\s*$/d' | wc -w", shell=True)
				
				comments = int(comments.decode().strip())
				words = int(words.decode().strip())
				
				totalReviewRequests += 1
				totalComments += comments
				totalWords += words
	return totalReviewRequests, totalComments, totalWords

	
def getConceptStats(markedPassagesCsvPath):
	annotations = 0
	concepts = {}
	with open(markedPassagesCsvPath, 'r') as fin:
		dcfin = csv.DictReader((x.replace('\0', '') for x in fin), #ignore NULL bytes, which crash the csv.DictReader
							  delimiter=';')
		for row in dcfin:
			annotations += 1
			conceptName = row['Category Title']
			concepts[conceptName] = concepts.get(conceptName, 0) + 1
			
	return len(concepts), annotations
	

if __name__ == '__main__':
	reviewRequests, comments, words = getBasicAnalysisStats('output/qcamap')
	print(f"Review requests: {reviewRequests}")
	print(f"Comments: {comments}")
	print(f"Words: {words}")
	
	concepts, annotations = getConceptStats('output/qcamap/markedpassages.csv')
	print(f"Concepts: {concepts}")
	print(f"Annotations: {annotations}")
	
		
			

#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import re

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
	

def getReviewRequestStats(basePath):
	repoNameRegex = re.compile('repository (.*?) at ')
	developerRegex = re.compile('by (.*?) ')
	
	repoFrequency = {}
	developerFrequency = {}
	
	for root, dir, files in os.walk(basePath):
		for f in files:
			if f.startswith('review') and f.endswith('txt'):
				filePath = os.path.join(root, f)
				with open(filePath, encoding='utf-16') as fin:
					for line in fin.readlines():
						if line.strip().startswith('$$$Review Request'):
							repo = repoNameRegex.search(line).group(1)
							repoFrequency[repo] = repoFrequency.get(repo, 0) + 1
							
						if line.strip().startswith('$$$'):
							regexResult = developerRegex.search(line)
							if regexResult:
								developer = regexResult.group(1)
								developerFrequency[developer] = developerFrequency.get(developer, 0) + 1

	return repoFrequency, developerFrequency
	

if __name__ == '__main__':
	reviewRequests, comments, words = getBasicAnalysisStats('output/qcamap')
	print(f"Review requests: {reviewRequests}")
	print(f"Comments: {comments}")
	print(f"Words: {words}")
	
	concepts, annotations = getConceptStats('output/qcamap/markedpassages.csv')
	print(f"Concepts: {concepts}")
	print(f"Annotations: {annotations}")
	
	repoFrequency, developerFrequency = getReviewRequestStats('output/qcamap')
	print(f"Repositories ({len(repoFrequency)}) and review requests:")
	for repo in sorted(repoFrequency):
		print(f"\t{repo}: {repoFrequency[repo]}")
	print(f"Developers ({len(developerFrequency)}) and comments:")
	for developer in sorted(developerFrequency):
		print(f"\t{developer}: {developerFrequency[developer]}")

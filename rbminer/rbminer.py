#!/usr/bin/env python3

import os
import csv
import requests
from rbtools.api.client import RBClient

REVIEWBOARD_BASE_URL = 'https://reviews.apache.org/'
DOWNLOADS_FOLDER = 'downloads'

def serializeField(fieldId, fieldContents):
	return '$$$%s\n%s\n\n' % (fieldId, fieldContents)
	

class DownloadCache(object):
	def __init__(self, basePath):
		self.path = os.path.join(basePath, 'cache')
		self.cache = None
		
	
	def readCacheFromDisk(self):
		self.cache = set()
		if os.path.exists(self.path):
			with open(os.path.join(self.path), 'r') as fin:
				for l in fin.readlines():
					if l.strip():
						self.cache.add(l.strip())
	
	
	def add(self, reviewId):
		if self.cache is None:
			self.readCacheFromDisk()
		
		with open(os.path.join(self.path), 'a') as fout:
			fout.write(str(reviewId) + '\n')
			
		self.cache.add(str(reviewId))
		
		
	def contains(self, reviewId):
		if self.cache is None:
			self.readCacheFromDisk()
		return str(reviewId) in self.cache


class ReplyBodyTop(object):
	def __init__(self, id_, text, username, timestamp):
		self.id_ = id_
		self.text = text
		self.username = username
		self.timestamp = timestamp
	
		
	def __str__(self):
		return serializeField('Review.reply.body_top, %s, %s, %s' % (self.id_, 
			self.username, self.timestamp), self.text)
			
	
	def asDict(self):
		return {'replyId': self.id_, 'type': 'Review.reply.body_top', 'text': self.text, 
			'username': self.username, 'timestamp': self.timestamp}
			

class ReplyBodyBottom(object):
	def __init__(self, id_, text, username, timestamp):
		self.id_ = id_
		self.text = text
		self.username = username
		self.timestamp = timestamp
	
		
	def __str__(self):
		return serializeField('Review.reply.body_bottom, %s, %s, %s' % (self.id_, 
			self.username, self.timestamp), self.text)
			
	
	def asDict(self):
		return {'replyId': self.id_, 'type': 'Review.reply.body_bottom', 'text': self.text, 
			'username': self.username, 'timestamp': self.timestamp}


class ReplyDiffComment(object):
	def __init__(self, id_, replyId, originalDiffCommentId, text, username, timestamp):
		self.id_ = id_
		self.replyId = replyId
		self.originalDiffCommentId = originalDiffCommentId
		self.text = text
		self.username = username
		self.timestamp = timestamp
	
		
	def __str__(self):
		return serializeField('ReplyDiffComment, %s, %s, %s' % (self.id_, 
			self.username, self.timestamp), self.text)
			
	
	def asDict(self):
		return {'replyDiffCommentId': self.id_, 'replyId': self.replyId, 
			'diffCommentId': self.originalDiffCommentId, 
			'type': 'ReplyDiffComment', 'text': self.text, 
			'username': self.username, 'timestamp': self.timestamp}
		

class DiffComment(object):
	def __init__(self, id_, text, username, timestamp):
		self.id_ = id_
		self.text = text
		self.username = username
		self.timestamp = timestamp
		
		self.replies = []


	def addReplyDiffComment(self, id_, replyId, text, username, timestamp):
		self.replies.append(ReplyDiffComment(id_, replyId, self.id_, text, username, timestamp))


	def __str__(self):
		s = serializeField('DiffComment, %s, %s, %s' % (self.id_, 
			self.username, self.timestamp), self.text)
		for r in self.replies:
			s += str(r)
		return s
		
		
	def asDictList(self):
		d = [{'diffCommentId': self.id_, 'type': 'DiffComment', 'text': self.text, 
			'username': self.username, 'timestamp': self.timestamp}]
		for r in self.replies:
			d.append(r.asDict())
		return d
		
		
class Review(object):
	def __init__(self, id_, bodyTop, bodyBottom, username, timestamp):
		self.id_ = id_
		self.bodyTop = bodyTop
		self.bodyBottom = bodyBottom
		self.username = username
		self.timestamp = timestamp
		self.replyBodiesTop = []
		self.replyBodiesBottom = []
		self.diffComments = {}
		
		
	def addReplyBodyTop(self, b):
		self.replyBodiesTop.append(b)
		
	
	def addReplyBodyBottom(self, b):
		self.replyBodiesBottom.append(b)
		
		
	def __str__(self):
		s = ''
		s += serializeField('Review.body_top, %s' % self.id_, self.bodyTop)
		for rbt in self.replyBodiesTop:
			s += serializeField('Review.reply.body_top', rbt)
		for d in self.diffComments.values():
			s += str(d)
		for rbb in self.replyBodiesBottom:
			s += serializeField('Review.reply.body_bottom', rbb)
		s += serializeField('Review.body_bottom, %s' % self.id_, self.bodyBottom)
		return s


	def _addReviewIdToDict(self, d):
		d.update({'reviewId': self.id_,})


	def asDictList(self):
		rows = []
		
		if self.bodyTop and self.bodyTop.strip():
			rows.append({'type': 'Review.body_top', 
				'reviewId': self.id_,
				'text': self.bodyTop,
				'username': self.username, 'timestamp': self.timestamp})
		
		for rbt in self.replyBodiesTop:
			dictRbt = rbt.asDict()
			self._addReviewIdToDict(dictRbt)
			rows.append(dictRbt)
			
		for d in self.diffComments.values():
			for diffDict in d.asDictList():
				self._addReviewIdToDict(diffDict)
				rows.append(diffDict)
		
		for rbb in self.replyBodiesBottom:
			dictRbb = rbb.asDict()
			self._addReviewIdToDict(dictRbb)
			rows.append(dictRbb)
				
		if self.bodyBottom and self.bodyBottom.strip():
			rows.append({'type': 'Review.body_bottom', 
				'reviewId': self.id_,
				'text': self.bodyBottom,
				'username': self.username, 'timestamp': self.timestamp})
				
		return rows

				
class ReviewRequest(object):
	CSV_FIELDS = ['reviewRequestId', 'repository', 'reviewRequestSubmitter', 'reviewId', 'diffCommentId', 'replyId', 
		'replyDiffCommentId', 'type',  'username', 'timestamp', 'text']
	
	def __init__(self, id_, repository, submitter):
		self.id_ = id_
		self.repository = repository
		self.submitter = submitter
		self.reviews = []
		
	
	def addReview(self, r):
		self.reviews.append(r)
		
		
	def _addReviewRequestInfoToDict(self, d):
		d.update({'reviewRequestId': self.id_, 'repository': self.repository,
			'reviewRequestSubmitter': self.submitter })

	
	def writeCSV(self, basePath):
		filePath = os.path.join(basePath, 'review-%d.csv' % (self.id_))
		with open(filePath, 'w', newline='') as fout:
			csvWriter = csv.DictWriter(fout, fieldnames=self.CSV_FIELDS)
			csvWriter.writeheader()
			for r in self.reviews:
				for d in r.asDictList():
					self._addReviewRequestInfoToDict(d)
					csvWriter.writerow(d)
					
	

def getUserId(diff):
	return diff.links._fields['user']['href'].split('/')[-2]
	

def getReplyToId(replyDiff):
	return int(replyDiff.links._fields['reply_to']['href'].split('/')[-2])
	

'''
Workaround for bug in ReviewBoard API. Some diff comments are not
returned when using the endpoint (/api/review-requests/%d/reviews/%d/diff-comments/)
even though there are reply diff comments pointing to them
(/api/review-requests/%d/reviews/%d/reply/%d/diff-comments/%d) 
'''
def getMissingDiffComment(reviewRequestId, reviewId, diffCommentId):
	r = requests.get('%sapi/review-requests/%d/reviews/%d/diff-comments/%d/' % (
		REVIEWBOARD_BASE_URL, reviewRequestId, reviewId, diffCommentId))
	diffJson = r.json()['diff_comment']
	return DiffComment(int(diffJson['id']), diffJson['text'], 
		diffJson['links']['user']['href'].split('/')[-2], 
		diffJson['timestamp'])


def downloadReviewRequest(rr):
	print('Downloading review request %s' % rr.id)

	repository = rr.links['repository']['title']
	submitter = rr.links['submitter']['title']
	outReviewRequest = ReviewRequest(rr.id, repository, submitter)

	for rev in rr.get_reviews():
		print('\tDownloading review %s' % rev.id)
		
		outReview = Review(rev.id, rev.body_top, rev.body_bottom, getUserId(rev), rev.timestamp)
		
		for diff in rev.get_diff_comments():
			print('\t\tDownloading diff %d' % diff.id)
			outReview.diffComments[diff.id] = DiffComment(diff.id, diff.text, getUserId(diff), diff.timestamp)
			
		for reply in rev.get_replies():
			if reply.body_top and reply.body_top.strip():
				outReview.addReplyBodyTop(ReplyBodyTop(reply.id, reply.body_top, getUserId(reply), reply.timestamp))
			if reply.body_bottom and reply.body_bottom.strip():
				outReview.addReplyBodyBottom(ReplyBodyBottom(reply.id, reply.body_bottom, getUserId(reply), reply.timestamp))
			for replyDiff in reply.get_diff_comments():
				originalDiffId = getReplyToId(replyDiff)
				print('\t\tDownloading reply %d to diff %d' % (replyDiff.id, originalDiffId))
				if originalDiffId not in outReview.diffComments:
					print('\t\tDownloading missing diff %d' % originalDiffId)
					outReview.diffComments[originalDiffId] = getMissingDiffComment(
						rr.id, rev.id, originalDiffId)
				outReview.diffComments[originalDiffId].addReplyDiffComment(replyDiff.id, reply.id, replyDiff.text, getUserId(replyDiff), replyDiff.timestamp)
				
		outReviewRequest.addReview(outReview)
		
	outReviewRequest.writeCSV(DOWNLOADS_FOLDER)
			
	
if __name__ == '__main__':
	client = RBClient(REVIEWBOARD_BASE_URL)
	root = client.get_root()
	
	if not os.path.exists(DOWNLOADS_FOLDER):
		os.makedirs(DOWNLOADS_FOLDER)
	
	downloadCache = DownloadCache(DOWNLOADS_FOLDER)
	
	
	for rid in range(72000, 0, -1):
		try:
			if not downloadCache.contains(rid):
				print('Checking if review request %s is valid' % rid)
				rr = root.get_review_request(review_request_id=rid)
				downloadReviewRequest(rr)
				downloadCache.add(rr.id)
		except:
			pass
	
	
	
	# ~ req = root.get_review_requests(last_updated_to='2019-09-01')
	# ~ for rr in req:
		# ~ if not downloadCache.contains(rr.id):
			# ~ downloadReviewRequest(rr)
			# ~ downloadCache.add(rr.id)
			
	
	
	# ~ rr = root.get_review_request(review_request_id=69623)
	# ~ if not downloadCache.contains(rr.id):
		# ~ downloadReviewRequest(rr)
		# ~ downloadCache.add(rr.id)

	# ~ req = root.get_review_requests(issue_open_count_gte=70)	
	# ~ for rr in req:
		# ~ if not downloadCache.contains(rr.id):
			# ~ downloadReviewRequest(rr)
			# ~ downloadCache.add(rr.id)


import io
import sys
import codecs
import argparse
import spice_wordnet as sw
import json
import numpy as np
import time
import os

import nltk
nltk.data.path.append('/media/Work_HD/yswang/nltk_data')
from nltk.corpus import wordnet

class SemanticTuple(object):

	def __init__(self, word):

		self.word = ' '.join(word.strip().lower().split())
		self.word_to_synset()


	def word_to_synset(self):

		lemma_synset = []
		word_split = self.word.split()
		if len(word_split) >= 2:
			self.word = "_".join(word_split)
		lemma_synset.append(self.word)

		for sys in wordnet.synsets(self.word):
			for l in sys.lemmas():
				lemma_synset.append(l.name())

		self.lemma_synset = set(lemma_synset)

def similar(tup_syns, pred):
	if len(tup_syns) != len(pred): 
		return False
	else:
		for w_id in range(len(tup_syns)):
			#print "w_id:  ", w_id
			
			if len(tup_syns[w_id].intersection(pred[w_id])) == 0:
				return False
		return True

def evaluate_spice(spice_tuple, ref_tuple):
	count_tuple = 0

	spice_predict_tuple = spice_tuple[:]
	num_ref   = len(ref_tuple)
	num_pred = len(spice_tuple)
	check_ref  = np.zeros((num_ref))
	check_pred = np.zeros((num_pred))

	ans = []
	for tup_id, tup in enumerate(ref_tuple):
		for spice_id, spice_tup in enumerate(spice_tuple):
			if check_pred[spice_id]==0 and tup==spice_tup:
				ans.append(tup)
				check_ref[tup_id] = 1
				check_pred[spice_id] = 1
				count_tuple += 1
				break		

	spice_wordnet = []

	for tup_id, tup in enumerate(spice_tuple):
		tup_syns = []
		if check_pred[tup_id] != 1:
			for word in tup:
				st = SemanticTuple(word)
				tup_syns.append(st.lemma_synset)

		spice_wordnet.append(tuple(tup_syns))
	

	for tup_id, tup in enumerate(ref_tuple):
		if check_ref[tup_id] == 1:
			continue
		tup_syns = []

		for word in tup:
			st = SemanticTuple(word)
			tup_syns.append(st.lemma_synset)

		for pred_id, pred in enumerate(spice_wordnet):
			if check_pred[pred_id]==0 and similar(tup_syns, pred):
				count_tuple += 1 
				check_ref[tup_id]   = 1
				check_pred[pred_id] = 1
				break
			

	if num_pred == 0:
		p_score = 0
	else:
		p_score = count_tuple/float(num_pred)

	s_score = count_tuple/float(num_ref)

	if count_tuple == 0:
		sg_score = 0
	else:
		sg_score = 2*p_score*s_score/(p_score+s_score)

	if sg_score > 1:
		#print ref_tuple
		#print spice_wordnet
		print len(ref_tuple)
		print len(spice_wordnet)
		print p_score
		print s_score
		print sg_score
		print "FUCK"
		exit()


	return sg_score

test = json.load(open('output_seb_dev.json', 'r'))

total_spice_tuple = []
for pred in test:
	spice_tuple = []
	for tup in pred['test_tuples']:
		if len(tup['tuple']) == 1:
			spice_tuple.append(tup['tuple'])
		else:
			spice_tuple.append(tuple(tup['tuple']))
	total_spice_tuple.append(spice_tuple)

test_sg = json.load(open('dev_sg.json', 'r'))

total_total_score = []
for ireg, reg in enumerate(total_spice_tuple):
	total_score = []
	t1 = time.time()
	print "No.: %d/%d" %(ireg, len(total_spice_tuple))
	for img in test_sg:
		s = 0
		
		for gt_reg in img:
			
			score = evaluate_spice(reg, gt_reg)
			s += score

		total_score.append(s/float(len(img)))

	t2 = time.time()
	print "cost time: ", t2-t1
	total_total_score.append(total_score)

json.dump(total_total_score, open('img_score_spice_dev.json', 'w'))

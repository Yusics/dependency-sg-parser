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

test = json.load(open('seb_pre_region_graph_dev.json', 'r'))

class RG(object):

	def __init__(self, objects, attributes, relationships):
		self.objects = objects
		self.attributes = attributes
		self.relationships = relationships
		self.graph_tuple = []
		self.convert_tuple()


	def convert_tuple(self):
		# objects -> tuple
		for i in range(len(self.objects)):
			temp_obj = self.objects[i]["names"][0]
			self.graph_tuple.append([temp_obj])

		#attributes -> tuple
		for i in range(len(self.attributes)):
			sub_id = self.attributes[i]['subject']
			subject = self.objects[sub_id]['names'][0]
			attribute = self.attributes[i]['attribute']
			self.graph_tuple.append(tuple([subject, attribute]))

		for i in range(len(self.relationships)):
			sub_id = self.relationships[i]['subject']
			obj_id = self.relationships[i]['object']
			sub = self.objects[sub_id]["names"][0]
			obj = self.objects[obj_id]["names"][0]
			self.graph_tuple.append(tuple([sub, self.relationships[i]['predicate'], obj]))


class Node:
	def __init__(self, id, word, parent_id, relation):
		self.id        = id
		self.word      = word
		self.parent_id = parent_id
		self.relation  = relation

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


#####create ground truth######
def isSubgraph(reg_sg, test_img_sg):

	isSub = True
	reg_graph = []
	for obj in reg_sg['objects']:
		reg_graph.append(obj)

	for attr_pair in reg_sg['attributes']:
		reg_graph.append(attr_pair)

	for rel in reg_sg['relations']:
		reg_graph.append(rel)

	for tup in reg_graph:
		
		#print test_img_sg
		
		if tup not in test_img_sg:
			isSub = False
			break
		

	return isSub

'''test_sg = []
gt      =[]

for i in range(len(test)):
	gt.append([])
	test_sg.append([])

#test_sg = np.array(test_sg)

for iimg, img, in enumerate(test):
	
	for ireg, reg in enumerate(img):
		test_sg[iimg].append([])
		for obj in reg['objects']:
			test_sg[iimg][ireg].append([obj])
		for attr_pair in reg['attributes']:
			test_sg[iimg][ireg].append(attr_pair)
		for rel in reg['relations']:
			test_sg[iimg][ireg].append(rel)



for iimg, img in enumerate(test):
	for ireg, reg in enumerate(img):
		gt[iimg].append([])
		for isg, sg in enumerate(test_sg):
			isSub = isSubgraph(reg, sg)
			if isSub:
				gt[iimg][ireg].append(isg)'''

####create ground truth finished #########
stanford_path = '/media/Work_HD/yswang/stanford-corenlp-full-2015-12-09/output_imgrt_dev'
stanford_tuple = []
for i in range(4953):
	file_name = 'output_json_'+str(i)+'.json'
	file = os.path.join(stanford_path, file_name)
	pred = json.load(open(file, 'r'))
	spice_tuple = RG(pred['objects'], pred['attributes'], pred['relationships']).graph_tuple
	stanford_tuple.append(spice_tuple)
#####
test_sg = json.load(open('dev_sg.json', 'r'))


total_total_score = []
for ireg, reg in enumerate(stanford_tuple):
	total_score = []
	t1 = time.time()
	print "No.: %d/%d" %(ireg, len(stanford_tuple))
	for img in test_sg:
		s = 0
		
		for gt_reg in img:
			#print gt_reg
			score = evaluate_spice(reg, gt_reg)
			s += score

		total_score.append(s/float(len(img)))

	t2 = time.time()
	print "cost time: ", t2-t1
	total_total_score.append(total_score)

json.dump(total_total_score, open('img_score_stanford_dev.json', 'w'))

		
	
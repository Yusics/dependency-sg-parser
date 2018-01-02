from optparse import OptionParser
from arc_hybrid_torch import ArcHybridLSTM
import pickle, utils, os, time, sys
import gc
import torch
import json
import numpy as np

import nltk
nltk.data.path.append('/media/Work_HD/yswang/nltk_data')
from nltk.corpus import wordnet

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


def find_tuples(node_list):
	tuples = []
	objects         = []
	objects_tail_id = []
	OBJTs_id = [] 

	attrs         = []
	attrs_tail_id = []

	preds         = []
	preds_tail_id = []

	
	for id_rnode, rnode in zip(range(len(node_list)-1,-1,-1), reversed(node_list)):
		if rnode.parent_id == -1:
			continue
		if rnode.relation == 'OBJT' or rnode.parent_id == 0:
			objects_tail_id.append(rnode.id)

			if rnode.relation == 'OBJT':
				if node_list[rnode.parent_id-1].relation == 'PRED':
					OBJTs_id.append(rnode.id)
			if rnode.id != 1 and node_list[id_rnode-1].relation == 'same':
				obj = [rnode.word]

				rnode_next = node_list[id_rnode-1]
				while True:
					obj.insert(0, rnode_next.word)
					if rnode_next.id == 1:
						break
					rnode_next = node_list[rnode_next.id-1-1]
					if rnode_next.relation != 'same':
						break
				objects.append(obj)
			else:
				objects.append([rnode.word])

	for id_rnode, rnode in zip(range(len(node_list)-1, -1, -1), reversed(node_list)):
		if rnode.relation == 'ATTR' or rnode.relation == 'PRED':
			if rnode.relation == 'ATTR':
				attrs_tail_id.append(rnode.id)
				if rnode.id != 1 and node_list[id_rnode-1].relation == 'same':
					attr = [rnode.word]
					rnode_next = node_list[id_rnode-1]
					while True:
						attr.insert(0, rnode_next.word)
						if rnode_next.id == 1:
							break
						rnode_next = node_list[rnode_next.id-1-1]
						if rnode_next.relation != 'same':
							break
					attrs.append(attr)
				else:
					attrs.append([rnode.word])

			else:
				preds_tail_id.append(rnode.id)
				if rnode.id != 1 and node_list[id_rnode-1].relation == 'same':
					pred = [rnode.word]
					rnode_next = node_list[id_rnode-1]
					while True:
						pred.insert(0, rnode_next.word)
						if rnode_next.id == 1:
							break
						rnode_next = node_list[rnode_next.id-1-1]
						if rnode_next.relation != 'same':
							break
					preds.append(pred)
				else:
					preds.append([rnode.word])

	for obj in objects:
		tuples.append([' '.join(obj)])

	#print preds_tail_id

	for attr_id, attr in enumerate(attrs):
		comp_attr = ' '.join(attr)
		obj_id = node_list[attrs_tail_id[attr_id]-1].parent_id
		if obj_id not in objects_tail_id:
			#print "attr object error"
			comp_obj = node_list[obj_id-1].word
		else:
			obj = objects_tail_id.index(obj_id)
			comp_obj = ' '.join(objects[obj])
		tuples.append((comp_obj, comp_attr))

	for OBJT_id in OBJTs_id:
		obj = objects_tail_id.index(OBJT_id)
		comp_obj = ' '.join(objects[obj])

		pred_id = node_list[OBJT_id-1].parent_id
		pred = preds_tail_id.index(pred_id)
		comp_pred = ' '.join(preds[pred])

		sub_id = node_list[pred_id-1].parent_id
		try:
			sub    = objects_tail_id.index(sub_id)
			comp_sub = ' '.join(objects[sub])
		except:
			comp_sub = node_list[sub_id-1].word
		
		tuples.append((comp_sub, comp_pred, comp_obj))

	return tuples


def sent_to_conll(sent_list):

	#total = []
	#print 'sentence list: ', sent_list
	total_sent = []
	sent_temp = []
	for sent in sent_list:
		
		ssent = sent.strip().split()
		for idx, word in enumerate(ssent):
			
			con_word = '%d\t%s\t_\t_\t_' % (idx+1 , word)
			sent_temp.append(con_word)

		sent_temp.append('')

	return sent_temp

def get_tuples(sent):

	node_list = []
	for word in sent:
		word = word.split('\t')
		#if word[2] == -1:
		#	continue
		try:
			node = Node(int(word[0]), word[1], int(word[2]), word[3])
		except:
			#print sent
			#print word
			node = Node(int(word[0]), word[1], -1, word[3])
		node_list.append(node)

	tuples = find_tuples(node_list)
	return tuples



def evaluate_spice(spice_tuple, ref_tuple):
	count_tuple = 0

	spice_predict_tuple = spice_tuple[:]
	num_ref   = len(ref_tuple)
	num_pred = len(spice_tuple)
	#print 'num ref: ', num_ref
	#print 'num pred: ', num_pred
	check_ref  = np.zeros((num_ref))
	check_pred = np.zeros((num_pred))

	ans = []
	#print 'ref tuple: ', ref_tuple
	#print 'pred tuple: ', spice_tuple
	for tup_id, tup in enumerate(ref_tuple):
		for spice_id, spice_tup in enumerate(spice_tuple):
			#print 'spice tup: ', spice_tup
			#print 'tup: ', tup
			if check_pred[spice_id]==0 and tup==spice_tup:
				#print 'I m in'
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

	if num_ref == 0:
		s_score = 0
	else:
		s_score = count_tuple/float(num_ref)

	#print 'count tuple: ', count_tuple
	#print 'p_score: ', p_score
	#print 's_score', s_score

	#s_score = count_tuple/float(num_ref)

	if count_tuple == 0:
		sg_score = 0
	else:
		if p_score+s_score == 0:
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

def read_conll(pred_conll, ref_conll, pred):
	#refs = json.load(open(gold_path, 'r'))
	#f    = codecs.open(conll_path, 'r', encoding='utf-8')

	#print 'PRED CONLL: ', len(pred_conll)
	#print len(f)
	#assert len(refs) == len(f)

	s_score = 0
	sent = []
	sent_refs = []
	count_gold = 0
	#print 'predict word conll', pred_conll
	for word in pred_conll:

		word = word.strip()
		if word == '':
			predict_tuples = get_tuples(sent)
			#ref_tuples     = sw.label_data(refs[count_gold])
			#print "PREDCIT tuples:\t", predict_tuples
			#print "REFERENCE tuples:\t", ref_tuples

			#spice_score    = evaluate_spice(predict_tuples, ref_tuples)
			#print spice_score
			#fout.write(str(spice_score)+'\n')
			#s_score += spice_score
			#count_gold += 1
			#sent = []

		else:
			sent.append(word)

	sent = []
	#fout = open()
	for word in ref_conll:
		if word == '':
			#predict_tuples = get_tuples(sent)
			ref_tuples = get_tuples(sent)
			#ref_tuples     = sw.label_data(refs[count_gold])
			
			#print 'ref_conll: ', sent
			spice_score    = evaluate_spice(predict_tuples, ref_tuples)
			print "PREDCIT tuples:\t", predict_tuples
			print "REFERENCE tuples:\t", ref_tuples
			print "SPICE score:\t", spice_score
			#print 'PRED', pred[count_gold]
			#exit()
			#id_to_score[idx].append(spice_score)
			#print 'spice score: ', spice_score
			#fout.write(str(spice_score)+'\n')
			s_score += spice_score
			count_gold += 1
			sent = []

		else:
			sent.append(word)

	#print 'count gold: ', count_gold
	#print 'number of reference: ',len(ref_conll)

	#assert count_gold == len(ref_conll)

	#print "Num of prediction: ", count_gold
	average_score = s_score/float(count_gold)
	#id_to_score[idx] = average_score

	return average_score


def main():

	
	'''pre_rgs = json.load(open('seb_pre_region_graph_dev.json', 'r'))
	sent = []
	for img in pre_rgs:
		for reg in img:
			sent.append(reg['phrase'])

	sent_conll = sent_to_conll(sent)
	with open('params.pickle', 'r') as paramsfp:
		words, w2i, rels, stored_opt = pickle.load(paramsfp)

		parser = ArcHybridLSTM(words, rels, w2i, stored_opt)
		parser.Load('barchybrid.model4_1e-2_256_200.tmp')
		pred = list(parser.Test(sent_conll))
		utils.write_conll('imgrt_dev.conll', pred)'''

	our_dev_tuples = []
	sent = []
	with open('imgrt_dev.conll', 'r') as fin:
		for line in fin.readlines():
			line = line.strip()
			if line == '':
				predict_tuples = get_tuples(sent)
				our_dev_tuples.append(predict_tuples)
				sent = []

			else:
				sent.append(line)

	json.dump(our_dev_tuples, open('our_dev_tuple.json', 'w'))

		




if __name__ == '__main__':
	main()
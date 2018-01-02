import io
import sys
import codecs
import spice_wordnet as sw
import json
import numpy as np
import time
import pickle
from optparse import OptionParser

import nltk
nltk.data.path.append('/media/Work_HD/yswang/nltk_data')
from nltk.corpus import wordnet



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

def readST_region_graphs(input_path):

	#key in jj[0]: relationships, url, height, regions, width, objects, attributes, id
	#region index start from 1, subject index start from 0

	with open(input_path, 'r') as ff:
		all_region_graphs = [json.loads(line) for line in ff]

	print all_region_graphs[0]['relationships'][2]
	print all_region_graphs[0]['attributes']
	print all_region_graphs[0]['objects']
	print all_region_graphs[0]['regions']
	
	return all_region_graphs

###preprocess
def process_labels(input_path):
	all_region_graphs = readST_region_graphs(input_path)
	total_region_graphs = []
	total_images = len(all_region_graphs)

	print "Total images: %d" % total_images

	for im in range(total_images):
		print "No.%d/%d" % (im, total_images)
		region_tuples = []
		region_graphs = all_region_graphs[im]
		for num_reg, reg in enumerate(region_graphs['regions']):
			region_dict = dict()
			phrase = region_graphs['regions'][num_reg]['phrase'].strip()
			region_dict['phrase']  = phrase[:len(phrase)-1]
			region_dict['attributes'] = []
			region_dict['relations'] = []
			region_dict['objects'] = []
			region_tuples.append(region_dict)

		for num_reg, attr_pair in enumerate(region_graphs['attributes']):
			region_tuples[attr_pair['region']-1]['attributes'].append((attr_pair['text'][0], attr_pair['text'][2]))
			if attr_pair['text'][0] not in region_tuples[attr_pair['region']-1]['objects']:
				region_tuples[attr_pair['region']-1]['objects'].append(attr_pair['text'][0])



		for rel in region_graphs['relationships']:
			region_tuples[rel['region']-1]['relations'].append(tuple(rel['text'])) 
			if rel['text'][0] not in region_tuples[rel['region']-1]['objects']:
				region_tuples[rel['region']-1]['objects'].append(rel['text'][0])
			if rel['text'][2] not in region_tuples[rel['region']-1]['objects']:
				region_tuples[rel['region']-1]['objects'].append(rel['text'][2])


		total_region_graphs.append(region_tuples)

	##process preprocessed labels
	test_sg = []

	for i in range(len(test)):
		#gt.append([])
		test_sg.append([])


	for iimg, img, in enumerate(test):
		
		for ireg, reg in enumerate(img):
			test_sg[iimg].append([])
			for obj in reg['objects']:
				test_sg[iimg][ireg].append([obj])
			for attr_pair in reg['attributes']:
				test_sg[iimg][ireg].append(attr_pair)
			for rel in reg['relations']:
				test_sg[iimg][ireg].append(rel)

			

	return total_region_graphs, test_sg

def read_pred(mode, pred_path):
	# 0: conll format
	# 1: Stanford format
	# 2: Spice format

	tuples = []
	if mode == 0:
		sent = []
		with open(pred_path, 'r') as fin:
			for line in fin.readlines():
				line = line.strip()
				if line == '':
					predict_tuples = get_tuples(sent)
					tuples.append(predict_tuples)
					sent = []

				else:
					sent.append(line)



	elif mode == 1:
		STANFORD_PRED = pred_path #'/media/Work_HD/yswang/stanford-corenlp-full-2015-12-09/output_imgrt_dev'
		for i in range(len([name for name in os.listdir(STANFORD_PRED) if os.path.isfile(name)])):
			file_name = 'output_json_'+str(i)+'.json'
			filepath = os.path.join(STANFORD_PRED, file_name)
			pred = json.load(open(filepath, 'r'))
			spice_tuple = RG(pred['objects'], pred['attributes'], pred['relationships']).graph_tuple
			tuples.append(spice_tuple)

	elif mode == 2:
		SPICE_PRED = json.load(open(pred_path, 'r'))
		for pred in test:
			spice_tuple = []
			for tup in pred['test_tuples']:
				if len(tup['tuple']) == 1:
					spice_tuple.append(tup['tuple'])
				else:
					spice_tuple.append(tuple(tup['tuple']))
			tuples.append(spice_tuple)






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


#####create wrong ground truth######
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
######################################

def cal_score(our_tuple, test_sg):

	total_total_score = []
	for ireg, reg in enumerate(our_tuple):
		total_score = []
		t1 = time.time()
		print "No.: %d/%d" %(ireg, len(our_tuple))
		for img in test_sg:
			s = 0
			
			for gt_reg in img:
				
				score = evaluate_spice(reg, gt_reg)
				s += score

			total_score.append(s/float(len(img)))

		t2 = time.time()
		print "cost time: ", t2-t1
		total_total_score.append(total_score)

	return total_total_score


def get_stat(sorted_idx, test_sg, mode):
	count = 0
	if mode == 0: #R@5
		
		for idx in sorted_idx[:5]:
			if idx in test_sg:
				count += 1
		return float(count)/len(test_sg)

	elif mode == 1:
		for idx in sorted_idx[:10]:
			if idx in test_sg:
				count += 1
		return float(count)/len(test_sg)



	elif mode == 2:
		for iidx, idx in enumerate(sorted_idx):
			if idx in test_sg:
				return iidx

parser = OptionParser()
parser.add_option("--input", dest="seb_input", help="Sebastian raw data", metavar="FILE", default="dev.processed.json")
parser.add_option("--mode", type="int", dest="mode", help="Mode for different format prediction\n 0: conll\n 1: Stanford\n 2: Spice", default=0)
parser.add_option("--pred", dest="pred_path", help="prediction path for different models", metavar="FILE", default="dev.conll")
parser.add_option("--gt", dest="gt", help="ground truth of data", metavar="FILE", default="gt.pkl")

(options, args) = parser.parse_args()


test, test_sg     = process_labels(options.seb_input)
pred              = read_pred(options.mode, options.pred)
total_total_score = cal_score(pred, test_sg)

gt = pickle.load(open(options.gt, 'r'))

count_rg  = 0
score_R5  = 0
score_R10 = 0
score_med = []

for gt_img in gt:
	for gt_reg in gt_img:
		unsort_scores = total_total_score[count_rg]
		sorted_idx = sorted(range(len(unsort_scores)), key=lambda k: unsort_scores[k], reverse=True)

		s_R5   = get_stat(sorted_idx, gt_reg, 0)
		s_R10  = get_stat(sorted_idx, gt_reg, 1)
		s_med  = get_stat(sorted_idx, gt_reg, 2)
	
		score_R5  += s_R5
		score_R10 += s_R10
		score_med.append(s_med)

		count_rg += 1

score_med = sorted(score_med)
for rank in score_med:
	print rank

print "score R@5: ", score_R5/float(len(total_total_score))
print "score R@10: ", score_R10/float(len(total_total_score))
print "score Med: ", score_med[len(score_med)/2] 




		
	
from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile
import codecs

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_PY = 'bist_coco.py'
TEMP_DIR = 'tmp'
#CACHE_DIR = 'cache'
 



class Spice:

	def compute_score(self, gts, res):

		assert(sorted(gts.keys()) == sorted(res.keys()))
		imgIds = sorted(gts.keys())

		# Prepare temp input file for the SPICE scorer
		input_data = []
		for idx in imgIds:
			hypo = res[idx]
			ref = gts[idx]

			# Sanity check.
			assert(type(hypo) is list)
			assert(len(hypo) == 1)
			assert(type(ref) is list)
			assert(len(ref) >= 1)

			input_data.append({
				"image_id" : idx,
				"test" : hypo[0],
				"refs" : ref
			})


		cwd = os.path.dirname(os.path.abspath(__file__))
		temp_dir=os.path.join(cwd, TEMP_DIR)
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
		json.dump(input_data, in_file, indent=2)

		in_file.close()


		out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
		out_file.close()
		#cache_dir=os.path.join(cwd, CACHE_DIR)
		#if not os.path.exists(cache_dir): os.makedirs(cache_dir)
		spice_cmd = ['python2', SPICE_PY,
					'--input', in_file.name,
					'--output', out_file.name,
					#'--output', 'result.json',
					'--model', 'barchybrid.model4_1e-2_256_200.tmp',
					'--params', 'params.pickle']


		subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
		#print 'os path: ', os.path.dirname(os.path.abspath(__file__))
		# Read and process results
		#print 'name of out_file 2: ', out_file.name
		#sys.exit()
		#exit()
		with open(out_file.name) as data_file:    
			result = json.load(data_file)
		#exit()

		os.remove(in_file.name)
		os.remove(out_file.name)

		imgId_to_scores = {}
		spice_scores = []
		for item in result:
			#print item['image_id']
			imgId_to_scores[item['image_id']] = item['scores']
			spice_scores.append(float(item['scores']['All']['f']))
		average_score = np.mean(np.array(spice_scores))
		scores = []
		for image_id in imgIds:
			# Convert none to NaN before saving scores over subcategories
			score_set = {}
			for category,score_tuple in imgId_to_scores[image_id].iteritems():
				score_set[category] = {k: float(v) for k, v in score_tuple.items()}
			scores.append(score_set)

		return average_score, scores

	def method(self):
		return "SPICE"


'''def main():
	# set up file names and pathes
	dataDir='.'
	dataType='val2014'
	algName = 'fakecap'
	annFile='captions_val2014.json'
	subtypes=['results', 'evalImgs', 'eval']
	[resFile, evalImgsFile, evalFile]= \
	['captions_val2014_fakecap_results.json' for subtype in subtypes]

	#gts_path = 'captions_val2014.json'
	#res_path = 'captions_val2014_fakecap_results.json'
	gts = json.load(open(annFile))
	res = json.load(open(resFile))
	print len(gts)
	#print gts[0]
	print len(res)
	
	print res[0] 
	exit()
	sp = Spice()
	score = sp.compute_score(gts, res)
	print score

if __name__ == '__main__':
	main()'''















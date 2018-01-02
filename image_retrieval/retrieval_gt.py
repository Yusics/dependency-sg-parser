import json
import pdb
import pickle
import sys
# from numpy import prod


def hasOverlap(l1, l2):
    return len([val for val in l1 if val in l2]) > 0


def dfs(aligns, candidate, complete_aligns):
    if len(aligns) == 0:
        complete_aligns.append(candidate)
    else:
        for align in aligns[0]:
            cand = [elem for elem in candidate]
            cand.append(align)
            dfs(aligns[1:], cand, complete_aligns)


def checkConsistent(complete_align):
    d = {}
    for align in complete_align:
        if len(align) == 2:
            if align[0] in d and d[align[0]] != align[1]:
                return False
            d[align[0]] = align[1]
        elif len(align) == 4:
            if (align[0] in d and d[align[0]] != align[1]) or (align[2] in d and d[align[2]] != align[3]):
                return False
            d[align[0]] = align[1]
            d[align[2]] = align[3]
        else:
            raise ValueError('unexpected len(align): {}'.format(len(align)))
    return True


def isSubgraph(src_img, reg_id, tar_img):
    candidate_aligns = []

    for src_attr in src_img['attributes']:
        if src_attr['region'] == reg_id:
            attr_align = []
            for tar_attr in tar_img['attributes']:
                if src_attr['attribute'] == tar_attr['attribute'] and hasOverlap(
                    src_img['objects'][src_attr['subject']]['names'], 
                    tar_img['objects'][tar_attr['subject']]['names']
                    ):
                    attr_align.append((src_attr['subject'], tar_attr['subject']))
            if len(attr_align) == 0:
                return False
            candidate_aligns.append(attr_align)

    for src_rel in src_img['relationships']:
        if src_rel['region'] == reg_id:
            rel_align = []
            for tar_rel in tar_img['relationships']:
                if src_rel['predicate'] == tar_rel['predicate'] and hasOverlap(
                    src_img['objects'][src_rel['subject']]['names'],
                    tar_img['objects'][tar_rel['subject']]['names']
                    ) and hasOverlap(
                    src_img['objects'][src_rel['object']]['names'],
                    tar_img['objects'][tar_rel['object']]['names']
                    ):
                    rel_align.append((src_rel['subject'], tar_rel['subject'], 
                        src_rel['object'], tar_rel['object']))
            if len(rel_align) == 0:
                return False
            candidate_aligns.append(rel_align)

    '''
    len_aligns = [len(elem) for elem in candidate_aligns]
    len_prod = prod(len_aligns)
    if len_prod > 100000:
        pdb.set_trace()
    '''

    complete_aligns = []
    dfs(candidate_aligns, [], complete_aligns)
    for complete_align in complete_aligns:
        if checkConsistent(complete_align):
            return True

    return False

#sys.argv[1]: path of Sebastian raw data
with open(sys.argv[1], 'r') as f:
    test = [json.loads(line) for line in f]

gt = []
count_reg, count_match = 0, 0
for iimg, img in enumerate(test):
    gt.append([])
    count_reg += len(img['regions'])
    for ireg, reg in enumerate(img['regions']):
        gt[iimg].append([iimg])
        for tar_iimg, tar_img in enumerate(test):
            if iimg == tar_iimg:
                continue
            if isSubgraph(img, ireg + 1, tar_img):
                gt[iimg][ireg].append(tar_iimg)
                continue
        print('image {} region {}: {} aligns'.format(
            iimg + 1, ireg + 1, len(gt[iimg][ireg])))
        count_match += len(gt[iimg][ireg])
        if len(gt[iimg][ireg]) > 5:
            print(reg['phrase'])

print('Total number of regions: {}'.format(count_reg))
print('Total number of matches: {}'.format(count_match))

#sys.argv[2] name of preprocessed data
with open(sys.argv[2], 'wb') as fp:
    pickle.dump(gt, fp)
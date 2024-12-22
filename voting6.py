import sys
import math
import numpy as np
import random

# for evaluation over separate test set:
seed = 12345
test_perm_file = "testlist.txt" #"testperm.txt"
test_size = 50 # 20 # in total
train_count = 500 # number of train subsets to evaluate
train_size = int(sys.argv[1]) # per class; 0 for full set (must be balanced), -1 for cross-validation, -2 all-train (print JmaxT)

log_file = sys.argv[2] # results file containing distances
# strategy parameters:
reduction_idx = int(sys.argv[3]) # 0: mean, 1: std, etc.
log_dist = sys.argv[4]=='1' # use logarithm
strategy = int(sys.argv[5]) # 0: baseline, 1-5: normal-fitting (clean, clean phi, mean, mean pre-phi, mean post-phi), 6-9: interpolation (clean, mean, 3pi)
share_stats = sys.argv[6]=='1' # share distance statistics among reference models
thresholding = int(sys.argv[7]) # -1: preset, 0: constant, 1: max J, 2: weighted mean of means, 3: normal fitting
exclude_val = sys.argv[8]=='1' # use cross-validation when determining scores on train for thresholding

# filtering parameters
if len(sys.argv)>=10:
	min_asr = float(sys.argv[9])-0.000001 # min ASR
	weak_file = sys.argv[10] # weak_attacks file
	strong_map = {}
	with open(weak_file) as f:
		lines = f.readlines()
		for line in lines:
			l = line.split(' ')
			if l[1]=='1':
				strong_map[l[0]]=float(l[5])>min_asr
else:
	strong_map = None

# preset threshold
if len(sys.argv)>=12:
	presetT = float(sys.argv[11])

def phi(x): # Cumulative distribution function for the standard normal distribution
	return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def cdf(x,data,up=True):
	if len(data)==0:
		return math.nan
	std = np.std(data)
	if std==0 and x==data[0]:
		return 0.5 if up else 0
	if not up:
		return (x-np.mean(data))/std
	if std==0:
		return int(x>data[0])
	return phi((x-np.mean(data))/std)

def aggregate(x,y): # 6-9: interpolation (clean, mean, 3pi)
	if strategy==7:
		return (x+y)/2
	if strategy>7:
		return x*y/(x*y+(1-x)*(1-y)) # "three pi" operator
	return x

def backdoor(x):
	if '.pth' in x:
		x = x[(x.find('_'+x.split('_')[0])+1):] # We rely on the fact that find() returns -1 if the argument is not found.
		return '-' in x.split('_')[0]
	return True

def to_exclude(x):
	if strong_map is None:
		return False
	return backdoor(x) and not strong_map[x]

def get_vote(ref,validation,train_clean_set,train_backdoor_set): # get vote of "ref" for "validation"
	def cec(x): # Clean Empirical CDF
		return np.mean([X<x for X in train_clean]) # "<" to avoid NaN in gaps with 3pi
	def bec(x): # Backdoor Empirical CDF
		return np.mean([X<=x for X in train_backdoor]) # "<=" to avoid NaN in gaps with 3pi
	d = dist_mat[ref][validation]
	if strategy==0:
		return d
	train_clean = []
	train_backdoor = []
	for r in train_clean_set if share_stats else [ref]:
		if r!=validation or not exclude_val:
			train_clean += [dist_mat[r][x] for x in train_clean_set if (x!=validation or not exclude_val) and x!=r]
			train_backdoor += [dist_mat[r][x] for x in train_backdoor_set if (x!=validation or not exclude_val) and x!=r]
	if strategy<6: # 1-5: normal-fitting (clean, clean phi, mean, mean pre-phi, mean post-phi)
		x = cdf(d,train_clean,strategy%2==0)
		if strategy>2:
			x = (x+cdf(d,train_backdoor,strategy%2==0))/2
		if strategy>4:
			x = phi(x)
		return x
	# 6-9: interpolation (clean, mean, 3pi)
	sl = sorted(train_clean+train_backdoor)
	index = sum(x<d for x in sl)
	if index==0:
		return 0
	if index==len(sl):
		return 1
	a = sl[index-1]
	b = sl[index]
	w = (d-a)/(b-a)
	if strategy<9:
		return (1-w)*aggregate(cec(a),bec(a))+w*aggregate(cec(b),bec(b))
	return aggregate((1-w)*cec(a)+w*cec(b),(1-w)*bec(a)+w*bec(b))

def get_score(validation,train_clean_set,train_backdoor_set): # get score for "validation"
	votes = []
	for ref in train_clean_set:
		if ref!=validation:
			votes.append(get_vote(ref,validation,train_clean_set,train_backdoor_set))
	if np.isnan(votes).any():
		raise RuntimeError('not enough data')
	if len(votes)%2==0:
		votes.append(-math.inf) # if the voting results in a draw, accept the model
	return np.median(votes)

def get_prediction(score):
	if thresholding==-1: # preset
		return presetT<score
	if thresholding==0: # constant
		return 0.5<score
	if thresholding==1: # max J
		return JmaxT<score
	if thresholding==2: # weighted mean of means
		return wmomT<score
	return 1<cdf(score,clean_scores)+cdf(score,backdoor_scores) # normal fitting

def get_predictions(labelled_scores,t=None):
	return [(get_prediction(ls[0]) if t is None else t<ls[0],ls[1]) for ls in labelled_scores]

def get_J(labelled_predictions):
	cm = [[0,0],[0,0]]
	for lp in labelled_predictions:
		cm[lp[1]][int(lp[0])] += 1
	FPR = cm[0][1]/sum(cm[0])
	FNR = cm[1][0]/sum(cm[1])
	J = 1-(FPR+FNR)
	return J

dist_mat = {}

with open(log_file) as f:
	lines = f.readlines()
	for line in lines:
		l = line.split(' RESULT: ')
		models = l[0].split(' ')
		if models[0]==models[1] or to_exclude(models[0]) or to_exclude(models[1]):
			continue
		dist = float(eval(l[1])[reduction_idx])
		if log_dist:
			dist = math.log(dist)
		for model in models:
			if model not in dist_mat:
				dist_mat[model] = {}
		dist_mat[models[1]][models[0]] = dist
		if models[1] not in dist_mat[models[0]]:
			dist_mat[models[0]][models[1]] = dist

crossval = train_size<0
print_JmaxT = train_size==-2 # all-train
all_models = [x for x in dist_mat]
clean_models = [x for x in all_models if not backdoor(x)]
backdoor_models = [x for x in all_models if backdoor(x)]
#if len(sys.argv)>=12:
#	print(len(backdoor_models))
#	exit()
if crossval:
	Plist = []
else:
	random.seed(seed)
	test_set = []
	with open(test_perm_file) as f:
		for i in range(test_size):
			test_set.append(f.readline().strip())
	train_clean_superset = [x for x in clean_models if x not in test_set]
	train_backdoor_superset = [x for x in backdoor_models if x not in test_set]
	assert len(train_clean_superset)+len(train_backdoor_superset)+len(test_set)==len(all_models)
	if train_size==0:
		train_size = len(train_clean_superset)
		assert train_size==len(train_backdoor_superset)
		train_count = 1
	#print("training models:",len(train_clean_superset)+len(train_backdoor_superset),"backdoored:",len(train_backdoor_superset))
	#print("testing models:",len(test_set),"backdoored:",len([x for x in test_set if backdoor(x)]))
	#print("train subset size per class:",train_size)
	Jlist = []

for experiment in range(len(all_models) if crossval else train_count):
	if crossval:
		test_model = None if print_JmaxT else all_models[experiment]
		train_clean_set = [x for x in clean_models if x!=test_model]
		train_backdoor_set = [x for x in backdoor_models if x!=test_model]
	else:
		train_clean_set = random.sample(train_clean_superset,train_size)
		train_backdoor_set = random.sample(train_backdoor_superset,train_size)
	train_set = train_clean_set + train_backdoor_set
	train_ls = [(get_score(validation,train_clean_set,train_backdoor_set),backdoor(validation)) for validation in train_set]
	
	if thresholding==1: # calc JmaxT
		sorted_scores = sorted(set([x[0] for x in train_ls]))
		Jmax = -2
		JmaxT = sorted_scores[0]
		for i in range(len(sorted_scores)-1):
			t = (sorted_scores[i]+sorted_scores[i+1])/2
			J = get_J(get_predictions(train_ls,t))
			if J>Jmax-0.000001:
				Jmax = J
				JmaxT = t
	
	if print_JmaxT:
		print(JmaxT,end="")
		exit()
	
	if thresholding>1: # calc wmomT
		clean_scores = [x[0] for x in train_ls if not x[1]]
		backdoor_scores = [x[0] for x in train_ls if x[1]]
		w_clean = np.std(backdoor_scores)
		w_backdoor = np.std(clean_scores)
		w = w_clean+w_backdoor
		wmomT = (np.mean(clean_scores)+np.mean(backdoor_scores))/2 if w==0 else w_clean/w*np.mean(clean_scores)+w_backdoor/w*np.mean(backdoor_scores)
	
	if crossval:
		p = bool(get_prediction(get_score(test_model,train_clean_set,train_backdoor_set)))
		gt = backdoor(test_model)
		ts = ["clean","poisoned"]
		print(test_model,ts[gt],ts[p],file=sys.stderr)
		Plist.append((p,gt))
	else:
		test_ls = [(get_score(validation,train_clean_set,train_backdoor_set),backdoor(validation)) for validation in test_set]
		#print(get_J(train_ls,JmaxT),get_J(test_ls,JmaxT))
		Jlist.append(get_J(get_predictions(test_ls)))

if crossval:
	print(get_J(Plist),end="")
else:
	#print(np.mean(Jlist),np.std(Jlist))
	print(np.mean(Jlist),end="")

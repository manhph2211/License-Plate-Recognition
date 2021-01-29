from deploy import demo
from local_utils import getPath
import numpy as np 
import sys
import os
import json

def predict(path_dic):
	results={}
	for k,v in path_dic.items():
		results[k]=[]
		print("Angle ",k)
		for i,path in enumerate(v):
			re=demo(path[1])
			print("IMAGE {0} - {1} ---------> {2}".format(i+1,path[0],re))
			results[k].append([path,re])

	return results

results=predict(getPath())

with open('results_0_15.json','w') as f:
	json.dump(results,f,indent=4)


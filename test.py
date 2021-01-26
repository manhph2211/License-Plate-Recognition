from deploy import demo
from local_utils import getPath
import numpy as np 
import sys
import os
def predict(path_dic):
	results={}
	for k,v in path_dic.items():
		results[k]=[]
		print("Angle ",k)
		for i,path in enumerate(v):
			print("IMAGE {0} --------- {1}".format(i+1,demo(path)))
			results[k].append(demo(path))

	return results

# results=predict(getPath())
# print("------------------------------")

# print(results)


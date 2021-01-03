import json

with open('./wpod-net.json','r') as f:
	data=json.load(f)
	#print(data)


with open('./testjson.json','w') as f:
	json.dump(data,f,indent=4)

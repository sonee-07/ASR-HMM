import pandas as pd
import numpy as np

df = pd.read_hdf("./features/mfcc/timit_train.hdf")
df_delta=pd.read_hdf("./features/mfcc_delta/timit_train.hdf")
df_dd=pd.read_hdf("./features/mfcc_delta_delta/timit_train.hdf")
print(df.head())
features = np.array(df["features"].tolist())
features_delta=np.array(df_delta["features"].tolist())
features_delta_labels=np.array(df_delta["labels"].tolist())
features_dd=np.array(df_dd["features"].tolist())
features_dd_labels=np.array(df_dd["labels"].tolist())
labels = np.array(df["labels"].tolist())
unique_labels=[]


for ite in labels:
	if(ite not in unique_labels):
		unique_labels.append(ite)

		
number_of_examples=len(labels)
detached={}
detached_d={}
detached_dd={}

for label in unique_labels:
	if(label==""):
		label="space"
	detached[label]=pd.DataFrame()
	detached_d[label]=pd.DataFrame()
	detached_dd[label]=pd.DataFrame()
for i in range(0,number_of_examples):
	if(labels[i]==""):
		labels[i]="space"
	if(features_delta_labels[i]==""):
		features_delta_labels[i]="space"
	if(features_dd_labels[i]==""):
		features_dd_labels[i]="space"
	s1=pd.Series(features[i])
	detached[labels[i]]=detached[labels[i]].append(s1,ignore_index=True)
	s2=pd.Series(features_delta[i])
	detached_d[features_delta_labels[i]] = detached_d[features_delta_labels[i]].append(s2,ignore_index=True)
	s3=pd.Series(features_dd[i])
	detached_dd[features_dd_labels[i]]=detached_dd[features_dd_labels[i]].append(s3,ignore_index=True)

for k in detached:
	fname=str(k)+"_train.hdf"
	detached[k].to_hdf("./features2/mfcc/"+fname, k)
	fname=str(k)+"_train.hdf"
	detached_d[k].to_hdf("./features2/mfcc_delta/"+fname,k)
	fname=str(k)+"_train.hdf"
	detached_dd[k].to_hdf("./features2/mfcc_delta_delta/"+fname,k)


f1=open("unique_labels.txt",'w')
for l in unique_labels:
	if(l==""):
		f1.write("space\n")
	else:
		f1.write(str(l)+"\n")
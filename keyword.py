import os, sys
# sys.path.insert(1, './baseline_methods')

import keyword_modules as km
import warnings

warnings.filterwarnings('ignore')

nlp = km.initilize_nlp_model()

path = str(os.getcwd() + '/files/')
#nlp, dataset = km.get_dataset(nlp, path=path)
nlp, dataset = km.get_dataset_from_excel(nlp, path=path, fname='202030_2096404419.xlsx', colname='متن مطلب')

for i in range(len(dataset)):
    print(f'=== {i+1} ===========')
    dataset[i].text = dataset[i].preprocess

keywords = km.TFIDF(dataset, nKeywords=5)
i=1
for key in keywords:
    t=''
    for k, v in key.items():
        t = t + f'({k}:{str(round(v*10000)/100)}), '

    print(f'==== {str(i)}th document keywords:  {t}')
    i=i+1




s=[]
for i in keywords:
  s.extend(i.keys())

# F=[]
# uniqs= list(set(s))
# for i in uniqs:
#     F.append(s.count(i))
#     if F[-1]>=500:
#         print(i,F[-1])  
      

F=[]
uniqs= list(set(s))
for i in uniqs:
    F.append(s.count(i))
    if F[-1]>=400:
        print(i,F[-1])  
        
    

# for i in s:
#   F.append(s.count(i))
  
  
  # for i in uniqs:
#   F.append(s.count(i))
#   if F[-1]>=500:
#     print(i,F[-1])  
    

# c=[]  
# uniqs= list(set(s))


      
      
    

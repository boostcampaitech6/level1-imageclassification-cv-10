import pandas as pd
import os

#submit csv path를 지정!!
submit_csv = '/data/ephemeral/home/input/eval/info.csv' #ex) '../data/submit.csv'

#Ensemble할 model이 모여있는 경로를 지정 (ex - 3개의 모델이 한 폴더에 위치)
path = '/data/ephemeral/home/level1-imageclassification-cv-10/output/ensemble/' #ex) './Ensemble/'
csv_list = [os.path.join(path, f) for  f in os.listdir(path)]
print(csv_list)
submit = pd.read_csv(f'{submit_csv}', index_col=False)

temp = pd.DataFrame()
for index, csv in enumerate(csv_list):
    temp[f'{index}'] = pd.read_csv(csv, index_col=False)['ans']

submit['ans'] = temp.mode(axis=1)[0].astype('int')
print(submit)

submit.to_csv(f'{path}voting.csv', index=False)
print('Done!!')
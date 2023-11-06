import jsonlines
import pandas as pd
import tqdm
import sys
import os

# params setting 
input_path = sys.argv[1]
print('*'*5, "input path:", input_path, '*'*5)
os.makedirs('./data', exist_ok=True)
input_path = input_path
output_path = './data/modify_submit.csv'


# modify
column_labels = ['id', 'text']
train_df = pd.DataFrame(columns=column_labels)
num = sum(1 for line in open(input_path))
progress = tqdm.tqdm(total=num)
with jsonlines.open(input_path) as reader:
    for obj in reader:
        id = obj['id']
        text = obj['maintext']
        text = 'summarize: '+text
        new_row = {'id':id, 'text':text}
        train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
        progress.update(1)
        
train_df.to_csv(output_path,index=False,encoding='utf8')
print('*'*5, 'modify input successful', '*'*5)
print('*'*5, f'save at {output_path}', '*'*5)

# test
# df = pd.read_csv(output_path)
# id = df['id'][0]
# text = df['text'][0]
# print(f'id: {id}')
# print(f'text: {text}')

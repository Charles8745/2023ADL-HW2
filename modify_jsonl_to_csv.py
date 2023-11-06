import jsonlines
import argparse
import pandas as pd
import tqdm

# params setting 
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", required=True, type=str, help="jsonl format")
parser.add_argument("-o", "--output_path", required=True, type=str, help=".csv format")
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

# modify
column_labels = ['id', 'text', 'summary']
train_df = pd.DataFrame(columns=column_labels)
num = sum(1 for line in open(input_path))
progress = tqdm.tqdm(total=num)
with jsonlines.open(input_path) as reader:
    for obj in reader:
        id = obj['id']
        text = obj['maintext']
        summary = obj['title']
        new_row = {'id':id, 'text':text, 'summary':summary}
        train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
        progress.update(1)
        
train_df.to_csv(output_path,index=False,encoding='utf8')

# test
df = pd.read_csv(output_path)
id = df['id'][0]
text = df['text'][0]
summary = df['summary'][0]
print(f'id:{id}')
print(f'text:{text}')
print(f'summary:{summary}')

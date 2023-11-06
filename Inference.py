import jsonlines
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

# params setting 
output_path = sys.argv[1]
print('*'*5, "output path:", output_path, '*'*5)
decode_strategy = 'beam'
checkpoint = 50000
param = 8
input_path = './data/modify_submit.csv'
output_path = output_path
model_path = f"./ADLHW2_beam_50k"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

set_seed(5487)

# limit cuda memory usage!! please comment when deploy
# print('*'*50)
# print('I confine the GPU ram usage')
# print('*'*50)
# torch.cuda.set_per_process_memory_fraction(0.33, 0)
# torch.cuda.empty_cache()

# functions and instance
class dataset(Dataset):
    def __init__(self, input_path):
        self.df = pd.read_csv(input_path, encoding='utf8')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id, text = row['id'], row['text']
        return id, text


def main(id, text, model, tokenizer):
    with torch.no_grad():
        text = list(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids
        inputs = inputs.to(device)

        # decode strategies
        if decode_strategy == 'greedy':
            outputs = model.generate(inputs, max_new_tokens=64)
        if decode_strategy == 'beam':
            outputs = model.generate(inputs, max_new_tokens=64, num_beams=param, early_stopping=True)
        if decode_strategy == 'temp':
            outputs = model.generate(inputs, max_new_tokens=64, do_sample=True, top_k=0, temperature=param)
        if decode_strategy == 'topk':
            outputs = model.generate(inputs, max_new_tokens=64, do_sample=True, top_k=param)
        if decode_strategy == 'topp':
            outputs = model.generate(inputs, max_new_tokens=64, do_sample=True, top_p=param, top_k=0)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # clean gpu usage
        torch.cuda.empty_cache()
        inputs = None
        outputs = None

    return summary, id.item()

# load model and public
test_dataset = dataset(input_path)
test_dataloader = DataLoader(test_dataset, batch_size=1)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)
model.eval()

# Inference
output_list = []
progress = tqdm(total=len(test_dataloader))

for id, text in test_dataloader:
    summary, id = main(id, text, model, tokenizer)
    output_list.append({'title':summary,'id':str(id)})
    progress.update(1)



# save as jsonl
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(output_list) 
print('-'*40, '\n', f'save at {output_path}')  

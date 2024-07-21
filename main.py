import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from torch.utils.data import Dataset, DataLoader;

import requests;

import numpy as np;
import pandas as pd;

from transformers import AutoTokenizer, AutoModel;


weight_file_location = "weights.pt";

device = torch.device("cuda" torch.cuda.is_available() else "cpu");
print(device);

df = pd.read_csv(weight_file_location);

model_name = "bert-base-uncased" if df.iloc[0]['model'] == 'BERT' else "FacebookAI/roberta-base";
test_file_location = df.iloc[0]['test'];

# importing tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name);
	
# importing model
class Classifier(nn.Module):
	def __init__(self, model_name, num_classes):
		super(Classifier, self).__init__();
        	self.model = AutoModel.from_pretrained(model_name);
        	self.dropout = nn.Dropout(p=0.2);
        	self.fc = nn.Linear(self.model.config.hidden_size, num_classes);	
	
	def forward(self,input_ids,attention_mask):
		output = self.model(input_ids=input_ids, attention_mask=attention_mask);
        	output = self.dropout(output.pooler_output);
        	output = self.fc(output);
        	return output;

model = Classifier(model_name,3);

# dataset creation
class CustomDataset(Dataset):
	def __init__(self,path_x,max_length):
		self.max_length = max_length;
		
		if not os.path.exists(path_x):
			raise FileNotFoundError(path_x);
		
		with open(path_x,'r') as file:
			self.x = f.readlines();
			
	def __len__(self):
		return len(self.x);
	
	def __getitem__(self,idx):
		x = self.x[idx];
		output = tokenizer(x,max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt');
		
		return {
			'input_ids' : output['input_ids'].flatten(),
			'attention_mask' : output['attention_mask'].flatten()
		};

# dataloader creation
test_dataset = Dataset(f"./Data/DATA/{test_file_location}",max_length);
test_dataloader = DataLoader(test_dataset,batch_size=8);


test = []
for batch in test_dataloader:
  model.eval();

  input_ids = batch['input_ids'].to(device);
  attention_mask = batch['attention_mask'].to(device);

  output = model(input_ids,attention_mask);
  y_pred = torch.argmax(output,dim=-1);
  test.append(y_pred);

tmp = [];
for batch in test:
    for line in batch:
        tmp.append(line.item());
        
id_to_label = {
    0:"passage",
    1:"phrase",
    2:"multi"
};

test_pred_label = [id_to_label[_] for _ in tmp];

data = {
    "id":range(len(test_pred_label)),
    "spoilerType":test_pred_label
};

df = pd.DataFrame(data);
df.to_csv("./output/output.csv",index=False);

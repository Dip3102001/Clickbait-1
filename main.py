import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from torch.utils.data import Dataset, DataLoader;

import gdown;

import numpy as np;
import pandas as pd;

import io;
import os;

from transformers import AutoTokenizer, AutoModel, BertModel, RobertaModel;

from tqdm import tqdm;

import gc;

pd.set_option('display.max_colwidth', None);

weight_file_location = "weights.pt";

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
device = torch.device("cpu");
print(device);

df = pd.read_csv(weight_file_location);

print(df['labels']);
print("Enter Label :-");
index = int(input("Waiting for Input.. "))


model_name = "bert-base-uncased" if df.iloc[index]['model'] == 'BERT' else "FacebookAI/roberta-base";
test_file_location = df.iloc[index]['test'];
url = df.iloc[index]['weight'];


#downloading file
def get_file(url):
	FILE_ID = url.split("/")[5];
	m_url = f"https://drive.google.com/uc?export=download&id={FILE_ID}";
	output_file = "output.weights.pt";
	
	gdown.download(m_url,output_file,quiet=False);
	
	
# importing tokenizer
print("Downloading Tokenizer..");
tokenizer = AutoTokenizer.from_pretrained(model_name);


def getClassifier(model_name):
	if model_name == "FacebookAI/roberta-base":
		class ROBERTaClassifier(nn.Module):
    			def __init__(self, model_name, num_classes):
        			super(ROBERTaClassifier, self).__init__();
        			self.bert = RobertaModel.from_pretrained(model_name);
        			self.dropout = nn.Dropout(p=0.2);
        			self.fc = nn.Linear(self.bert.config.hidden_size, num_classes);

    			def forward(self, input_ids, attention_mask):
        			output = self.bert(input_ids=input_ids, attention_mask=attention_mask);
        			output = self.dropout(output.pooler_output);
        			output = self.fc(output);
        			return output;
	
		return ROBERTaClassifier(model_name,3);
	else:
		class Classifier(nn.Module):
			def __init__(self, model_name, num_classes):
				super(Classifier, self).__init__();
				self.bert = BertModel.from_pretrained(model_name);
				self.dropout = nn.Dropout(p=0.2);
				self.fc = nn.Linear(self.bert.config.hidden_size, num_classes);	
	
			def forward(self,input_ids,attention_mask):
				output = self.bert(input_ids=input_ids, attention_mask=attention_mask);
				output = self.dropout(output.pooler_output);
				output = self.fc(output);
				return output;
		
		return Classifier(model_name,3);

	
model = getClassifier(model_name);
model = model.to(device);
get_file(url);

model.load_state_dict(torch.load("output.weights.pt",map_location=device));

# dataset creation
class CustomDataset(Dataset):
	def __init__(self,path_x,max_length):
		self.max_length = max_length;
		
		if not os.path.exists(path_x):
			raise FileNotFoundError(path_x);
		
		with open(path_x,'r') as file:
			self.x = file.readlines();
			
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
max_length = 512;
test_dataset = CustomDataset(f"./Data/DATA/{test_file_location}",max_length);
test_dataloader = DataLoader(test_dataset,batch_size=2);


test = [];

model.eval();
for batch in tqdm(test_dataloader,ascii=True,desc="Processing..."):
  input_ids = batch['input_ids'].to(device);
  attention_mask = batch['attention_mask'].to(device);

  output = model(input_ids,attention_mask);
  y_pred = torch.argmax(output,dim=-1);
  test.append(y_pred);
  
  gc.collect();


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

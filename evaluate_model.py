import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

dataset_name = "twitter"
load_model_name =  "saved_models/model.pth"

# load_data
df_source = pd.read_csv(f'processed_dataset/test_source_{dataset_name}.csv')
df_reply = pd.read_csv(f'processed_dataset/test_reply_{dataset_name}.csv')
df_source['SourceKey'] = None
df_test = pd.concat([df_source, df_reply], ignore_index=True)

# load_model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load(load_model_name))
model.eval()

texts = df_test['Text_x'].tolist()
true_labels = df_test['Label'].tolist()
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True) for text in texts]
attention_masks = [[float(i>0) for i in seq] for seq in tokenized_texts]

input_ids = torch.tensor(tokenized_texts)
attention_masks = torch.tensor(attention_masks)
true_labels = torch.tensor(true_labels)

batch_size = 32
dataset = TensorDataset(input_ids, attention_masks, true_labels)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

# predict
predictions = []
for batch in dataloader:
    batch = tuple(t.to('cuda') for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0]
    
    predictions.append(logits.detach().cpu().numpy())


predictions = np.concatenate(predictions, axis=0)
predicted_labels = np.argmax(predictions, axis=1)

print(classification_report(true_labels, predicted_labels))

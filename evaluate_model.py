import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = "twitter"
load_model_name = "saved_models/best_reply_redit_with_context.pth"

# Load data
df_source = pd.read_csv(f'processed_dataset/test_source_{dataset_name}.csv')
df_reply = pd.read_csv(f'processed_dataset/test_reply_{dataset_name}.csv')
df_source['SourceKey'] = None
df_test = pd.concat([df_source, df_reply], ignore_index=True)

# Load model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)
model.load_state_dict(torch.load(load_model_name))
model.eval()

# Tokenize texts
texts = df_test['Text'].tolist()
true_labels = df_test['Label'].tolist()
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True) for text in texts]

# Pad sequences
max_len = 512  # max length for BERT
padded_texts = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in tokenized_texts])
attention_masks = np.where(padded_texts != 0, 1, 0)

# Convert lists to tensors
input_ids = torch.tensor(padded_texts)
attention_masks = torch.tensor(attention_masks)
true_labels = torch.tensor(true_labels)

# Create DataLoader
batch_size = 32
dataset = TensorDataset(input_ids, attention_masks, true_labels)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

# Predict
predictions = []
for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0]
    
    predictions.append(logits.detach().cpu().numpy())

# Concatenate predictions
predictions = np.concatenate(predictions, axis=0)
predicted_labels = np.argmax(predictions, axis=1)

# Print classification report
print(classification_report(true_labels, predicted_labels))

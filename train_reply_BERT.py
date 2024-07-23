import torch
import time
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# merging context
df_train_reply = pd.read_csv('processed_dataset/reply_twitter.csv')
df_valid_reply = pd.read_csv('processed_dataset/reply_valid.csv')
df_train_pred = pd.read_csv('train_pred.txt', sep='\t')
df_valid_pred = pd.read_csv('valid_pred.txt', sep='\t')

df_train_reply['SourceKey'] = df_train_reply['SourceKey'].astype(str)  # set keys to string
df_valid_reply['SourceKey'] = df_valid_reply['SourceKey'].astype(str)
df_train_pred['Key'] = df_train_pred['Key'].astype(str)
df_valid_pred['Key'] = df_valid_pred['Key'].astype(str)

df_train = pd.merge(df_train_reply, df_train_pred, left_on='SourceKey', right_on='Key', how='inner')
df_valid = pd.merge(df_valid_reply, df_valid_pred, left_on='SourceKey', right_on='Key', how='inner')
print(df_train.head())
print(df_train['Label_x'].unique())
df_train['combined_text'] = df_train['Text_x'] + " [LABEL] " + df_train['Prediction'].astype(str) + " [SEP] " + df_train['Text_y']
df_valid['combined_text'] = df_valid['Text_x'] + " [LABEL] " + df_valid['Prediction'].astype(str) + " [SEP] " + df_valid['Text_y']

def create_loaders(df_train, df_valid, tokenizer, batch_size):
    tokenized_train = df_train['combined_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    tokenized_valid = df_valid['combined_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    sum1, sum2 = 0, 0
    nes = 0
    for i in tokenized_train:
        nes = max(nes, len(i))
        sum1 += len(i)
        if len(i) > 512:
            sum2 += 1
    print(nes)
    print(sum1/(len(tokenized_train)))
    print(sum2/(len(tokenized_train)))
    max_len = max([len(sen) for sen in pd.concat([tokenized_train, tokenized_valid], axis = 0)])
    padded_train = np.array([i + [0]*(max_len-len(i)) for i in tokenized_train])
    padded_valid = np.array([i + [0]*(max_len-len(i)) for i in tokenized_valid])

    attention_train = np.where(padded_train != 0, 1, 0)
    attention_valid = np.where(padded_valid != 0, 1, 0)

    train_inputs, valid_inputs = padded_train, padded_valid
    train_labels, valid_labels = df_train['Label_x'].values, df_valid['Label_x'].values
    train_masks, valid_masks = attention_train, attention_valid

    train_inputs, validation_inputs = torch.tensor(train_inputs), torch.tensor(valid_inputs)
    train_labels, validation_labels = torch.tensor(train_labels, dtype=torch.long), torch.tensor(valid_labels, dtype=torch.long)
    train_masks, validation_masks = torch.tensor(train_masks), torch.tensor(valid_masks)

    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    train_masks = train_masks.to(device)
    validation_inputs = validation_inputs.to(device)
    validation_labels = validation_labels.to(device)
    validation_masks = validation_masks.to(device)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(accuracy_score(labels_flat, pred_flat))
    print(recall_score(labels_flat, pred_flat))
    print(precision_score(labels_flat, pred_flat))
    print(f1_score(labels_flat, pred_flat))
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)
train_dataloader, validation_dataloader = create_loaders(df_train, df_valid, tokenizer, 32)

epochs = 150
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

set_seeds(42)
loss_values = []
t0 = time.time()

for epoch_i in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        #if step % 10 == 0 and not step == 0:
        #    elapsed = time.time() - t0
        #    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    
    loss_values.append(avg_train_loss)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    #print("  Training epoch took: {:}".format(time.time() - t0))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    #print("  Validation took: {:}".format(time.time() - t0))
print("")
print("Training complete!")

# predicting on training and validation data
def make_predictions(data_loader):
    predictions = []
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.detach().cpu().numpy())
    return predictions

df_train['Prediction_x'] = make_predictions(train_dataloader)
df_valid['Prediction_x'] = make_predictions(validation_dataloader)

print("Train Accuracy: ", accuracy_score(df_train['Label_x'], df_train['Prediction_x']))
print("Validation Accuracy: ", accuracy_score(df_valid['Label_x'], df_valid['Prediction_x']))

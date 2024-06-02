import torch
import time
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

def tokenize(df_train, df_valid, df_test):
    tokenized_train = df_train['Text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    tokenized_valid = df_valid['Text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    tokenized_test = df_test['Text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    max_len = max([len(sen) for sen in pd.concat([tokenized_train, tokenized_valid, tokenized_test], axis = 0)])
    return tokenized_train, tokenized_valid, tokenized_test, max_len

def create_iml(df, tokenized, max_len):
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
    attention = np.where(padded != 0, 1, 0)

    inputs = padded
    labels = df['Label'].values
    masks = attention

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(masks)

    inputs = inputs.to(device)
    labels = labels.to(device)
    masks = masks.to(device)

    return inputs, masks, labels

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seeds(42)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_name = 'twitter'
df_train = pd.read_csv(f'processed_dataset/source_{dataset_name}.csv')
df_valid = pd.read_csv(f'processed_dataset/source_valid.csv')
df_test = pd.read_csv(f'processed_dataset/test_source_{dataset_name}.csv')

train_tokenized, valid_tokenized, test_tokenized, max_len = tokenize(df_train, df_valid, df_test)
train_iml = create_iml(df_train, train_tokenized, max_len)
valid_iml = create_iml(df_valid, valid_tokenized, max_len)
test_iml = create_iml(df_test, test_tokenized, max_len)

train_data = TensorDataset(*train_iml)
train_sampler = RandomSampler(train_data)
validation_data = TensorDataset(*valid_iml)
validation_sampler = SequentialSampler(validation_data)
test_data = TensorDataset(*test_iml)
test_sampler = SequentialSampler(test_data)


best_accuracy = 0.0
best_model = None
best_hyperparameters = {}

batch_sizes = [16, 32, 64]
learning_rate = [2e-5, 2e-4, 2e-3]
num_epochs = [2, 3, 4]

for batch_size in batch_sizes:
    for lr in learning_rate:
        for epochs in num_epochs:
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4,
                output_attentions=False,
                output_hidden_states=False,
            )
            model.to(device)

            train_dataloader = DataLoader(
                train_data,
                sampler = train_sampler,
                batch_size = batch_size
            )
            validation_dataloader = DataLoader(
                validation_data,
                sampler = validation_sampler,
                batch_size = batch_size
            )
            test_dataloader = DataLoader(
                test_data,
                sampler = test_sampler,
                batch_size = batch_size
            )

            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
            total_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            loss_values = []
            t0 = time.time()

            for epoch_i in range(epochs):
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')

                total_loss = 0
                model.train()

                for step, batch in enumerate(train_dataloader):
                    if step % 5 == 0 and not step == 0:
                        elapsed = time.time() - t0
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

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

                print("")
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

                avg_eval_accuracy = eval_accuracy/nb_eval_steps
                print("  Accuracy: {0:.2f}".format(avg_eval_accuracy))
                
                if avg_eval_accuracy > best_accuracy:
                    best_accuracy = avg_eval_accuracy
                    best_model = model.state_dict()
                    best_hyperparameters = {
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'num_epochs': epochs
                    }
                    torch.save(best_model, f'./saved_models/best_source_{dataset_name}.pth')

            print("")
            print("Training complete!")

print(best_hyperparameters)

best_model_instance = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)
best_model_instance.load_state_dict(best_model)
best_model_instance.to(device)

# predicting on data
def make_predictions(data_loader):
    predictions = []
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = best_model_instance(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.detach().cpu().numpy())
    return predictions

df_train['Prediction'] = make_predictions(DataLoader(train_data, sampler = train_sampler, batch_size = 64))
df_valid['Prediction'] = make_predictions(DataLoader(validation_data, sampler = validation_sampler, batch_size = 64))
df_test['Prediction'] = make_predictions(DataLoader(test_data, sampler = test_sampler, batch_size = 64))

df_train.to_csv('train_pred.txt', sep='\t', index=False)
df_valid.to_csv('valid_pred.txt', sep='\t', index=False)
df_test.to_csv('test_pred.txt', sep='\t', index=False)

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
from sklearn.metrics import classification_report
from sklearn.utils import resample


def merge_datasets(dataset_name):
    df_train_reply = pd.read_csv(f'processed_dataset/reply_{dataset_name}.csv')
    df_valid_reply = pd.read_csv('processed_dataset/reply_valid.csv')
    df_test_reply = pd.read_csv(f'processed_dataset/test_reply_{dataset_name}.csv')
    df_train_pred = pd.read_csv(f'train_pred_{dataset_name}.txt', sep='\t')
    df_valid_pred = pd.read_csv('valid_pred.txt', sep='\t')
    df_test_pred = pd.read_csv(f'test_pred_{dataset_name}.txt', sep='\t')

    df_train_reply['SourceKey'] = df_train_reply['SourceKey'].astype(str)
    df_valid_reply['SourceKey'] = df_valid_reply['SourceKey'].astype(str)
    df_test_reply['SourceKey'] = df_test_reply['SourceKey'].astype(str)
    df_train_pred['Key'] = df_train_pred['Key'].astype(str)
    df_valid_pred['Key'] = df_valid_pred['Key'].astype(str)
    df_test_pred['Key'] = df_test_pred['Key'].astype(str)

    df_train = pd.merge(df_train_reply, df_train_pred, left_on='SourceKey', right_on='Key', how='inner')
    df_valid = pd.merge(df_valid_reply, df_valid_pred, left_on='SourceKey', right_on='Key', how='inner')
    df_test = pd.merge(df_test_reply, df_test_pred, left_on='SourceKey', right_on='Key', how='inner')
    df_train['combined_text'] = df_train['Text_x'] + " [LABEL] " + df_train['Prediction'].astype(str) + " [SEP] " + df_train['Text_y']
    df_valid['combined_text'] = df_valid['Text_x'] + " [LABEL] " + df_valid['Prediction'].astype(str) + " [SEP] " + df_valid['Text_y']
    df_test['combined_text'] = df_test['Text_x'] + " [LABEL] " + df_test['Prediction'].astype(str) + " [SEP] " + df_test['Text_y']

    return df_train, df_valid, df_test

def tokenize(df_train, df_valid, df_test):  # change no context or context the column: Text_x, combined_text
    tokenized_train = df_train['combined_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))
    tokenized_valid = df_valid['combined_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))
    tokenized_test = df_test['combined_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))
    max_len = max([len(sen) for sen in pd.concat([tokenized_train, tokenized_valid, tokenized_test], axis = 0)])
    return tokenized_train, tokenized_valid, tokenized_test, max_len

def create_iml(df, tokenized, max_len):
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
    attention = np.where(padded != 0, 1, 0)

    inputs = padded
    labels = df['Label_x'].values
    masks = attention

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(masks)

    inputs = inputs.to(device)
    labels = labels.to(device)
    masks = masks.to(device)

    return inputs, masks, labels


def create_data_samples(df_train, df_valid, df_test):
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

    return train_data, train_sampler, validation_data, validation_sampler, test_data, test_sampler

def flat_accuracy(preds, labels, show=False):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    if show:
        print("acc: ", accuracy_score(labels_flat, pred_flat), "\n",
              "P: ", precision_score(labels_flat, pred_flat, average='weighted', zero_division=0), "\n",
              "R: ", recall_score(labels_flat, pred_flat, average='weighted', zero_division=0), "\n",
              "F1: ", f1_score(labels_flat, pred_flat, average='weighted', zero_division=0))
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
set_seeds(seed)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_name = 'redit'
dataset_name_2 = 'twitter'
df_train, df_valid, df_test = merge_datasets(dataset_name)
#df_train_2, _, df_test_2 = merge_datasets(dataset_name_2)
if dataset_name_2 is not None:
    df_train_2, _, df_test_2 = merge_datasets(dataset_name_2)
    df_train = pd.concat([df_train, df_train_2], ignore_index=True)
    df_test = pd.concat([df_test, df_test_2], ignore_index=True)

def upsample(df):
    df_majority = df[df.Label_x == 3]  # comments
    df_minority1 = df[df.Label_x == 0]
    df_minority2 = df[df.Label_x == 1]
    df_minority3 = df[df.Label_x == 2]

    df_minority1_upsampled = resample(df_minority1, replace=True, n_samples=len(df_majority), random_state=seed)
    df_minority2_upsampled = resample(df_minority2, replace=True, n_samples=len(df_majority), random_state=seed)
    df_minority3_upsampled = resample(df_minority3, replace=True, n_samples=len(df_majority), random_state=seed)

    return pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled, df_minority3_upsampled])

train_data, train_sampler, validation_data, validation_sampler, test_data, test_sampler = create_data_samples(upsample(df_train), df_valid, upsample(df_test))
train_data_2, train_sampler_2, validation_data_2, validation_sampler_2, test_data_2, test_sampler_2 = create_data_samples(df_train, df_valid, df_test)

best_accuracy = 0.0
best_model = None
best_hyperparameters = {}
model = None
batch_sizes = [32]
learning_rate = [2e-5]
num_epochs = [5]

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
                    torch.save(best_model, f'./saved_models/best_reply_{dataset_name}_without_context.pth')

            print("")
            print("Training complete!")

print(best_hyperparameters)

# best_model_instance = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=4,
#     output_attentions=False,
#     output_hidden_states=False,
# )
# best_model_instance.load_state_dict(best_model)
# best_model_instance.to(device)

# predicting on data
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

train_dataloader_2 = DataLoader(
    train_data_2,
    sampler = train_sampler_2,
    batch_size = batch_size
)
validation_dataloader_2 = DataLoader(
    validation_data_2,
    sampler = validation_sampler_2,
    batch_size = batch_size
)
test_dataloader_2 = DataLoader(
    test_data_2,
    sampler = test_sampler_2,
    batch_size = batch_size
)

df_train['Prediction_x'] = make_predictions(train_dataloader_2)
df_valid['Prediction_x'] = make_predictions(validation_dataloader_2)
df_test['Prediction_x'] = make_predictions(test_dataloader_2)


print("Train report:\n", classification_report(df_train['Label_x'], df_train['Prediction_x']))
print("Validation report:\n", classification_report(df_valid['Label_x'], df_valid['Prediction_x']))
print("Test report:\n", classification_report(df_test['Label_x'], df_test['Prediction_x']))

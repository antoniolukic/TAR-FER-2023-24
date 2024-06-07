import time
import pandas as pd
import re
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def merge_datasets(dataset_name):
    df_train_reply = pd.read_csv(f'processed_dataset/reply_{dataset_name}.csv')
    df_valid_reply = pd.read_csv('processed_dataset/reply_valid.csv')
    df_test_reply = pd.read_csv(f'processed_dataset/test_reply_{dataset_name}.csv')
    df_train_pred = pd.read_csv('train_pred.txt', sep='\t')
    df_valid_pred = pd.read_csv('valid_pred.txt', sep='\t')
    df_test_pred = pd.read_csv('test_pred.txt', sep='\t')

    df_train_reply['SourceKey'] = df_train_reply['SourceKey'].astype(str)
    df_valid_reply['SourceKey'] = df_valid_reply['SourceKey'].astype(str)
    df_test_reply['SourceKey'] = df_test_reply['SourceKey'].astype(str)
    df_train_pred['Key'] = df_train_pred['Key'].astype(str)
    df_valid_pred['Key'] = df_valid_pred['Key'].astype(str)
    df_test_pred['Key'] = df_test_pred['Key'].astype(str)

    df_train = pd.merge(df_train_reply, df_train_pred, left_on='SourceKey', right_on='Key', how='inner')
    df_valid = pd.merge(df_valid_reply, df_valid_pred, left_on='SourceKey', right_on='Key', how='inner')
    df_test = pd.merge(df_test_reply, df_test_pred, left_on='SourceKey', right_on='Key', how='inner')
    df_train['combined_text'] = df_train['Text_x']  +  " [SEP] " + df_train['Text_y']
    df_valid['combined_text'] = df_valid['Text_x']  + " [SEP] " + df_valid['Text_y']
    df_test['combined_text'] = df_test['Text_x']  + " [SEP] " + df_test['Text_y']

    return df_train, df_valid, df_test

def tokenize(df_train, df_valid, df_test):
    df_train = df_train['combined_text'].apply(lambda x: process_text(x))
    df_valid = df_valid['combined_text'].apply(lambda x: process_text(x))
    df_test = df_test['combined_text'].apply(lambda x: process_text(x))

    tokenizer.fit_on_texts(list(df_train))
    df_train = tokenizer.texts_to_sequences(df_train)
    df_valid = tokenizer.texts_to_sequences(df_valid)
    df_test = tokenizer.texts_to_sequences(df_test)

    df_train = pad_sequences(df_train, maxlen=maxlen)
    df_valid = pad_sequences(df_valid, maxlen=maxlen)
    df_test = pad_sequences(df_test, maxlen=maxlen)
    # max_len = max([len(sen) for sen in pd.concat([df_train, df_test], axis = 0)])

    return df_train, df_test, df_valid  # , max_len

def process_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()

    return text

def load_glove(word_index):
    EMBEDDING_FILE = 'glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 750 # max number of words in a question to use
batch_size = 512 # how many samples to process at once
n_epochs = 5 # how many times to iterate over all samples
n_splits = 5 # Number of K-fold Splits
SEED = 10
debug = 0



dataset_name = 'twitter'
df_train, df_valid, df_test = merge_datasets(dataset_name)
tokenizer = Tokenizer(num_words=max_features)

train_tokenized, test_tokenized, valid_tokenized = tokenize(df_train, df_valid, df_test)

le = LabelEncoder()
train_y = le.fit_transform(df_train['Label_x'].values)
test_y = le.transform(df_test['Label_x'].values)
valid_y = le.transform(df_valid['Label_x'].values)


if debug:
    embedding_matrix = np.random.randn(120000,300)
else:
    embedding_matrix = load_glove(tokenizer.word_index)


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        # rint(x.size())
        h_embedding = self.embedding(x)
        # _embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

def flat_accuracy(preds, labels, show=False):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    if show:
        print("acc: ", accuracy_score(labels_flat, pred_flat), "\n",
              "P: ", precision_score(labels_flat, pred_flat, average='weighted', zero_division=0), "\n",
              "R: ", recall_score(labels_flat, pred_flat, average='weighted', zero_division=0), "\n",
              "F1: ", f1_score(labels_flat, pred_flat, average='weighted', zero_division=0))
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


n_epochs = 6
model = BiLSTM()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load train and test in CUDA Memory
x_train = torch.tensor(train_tokenized, dtype=torch.long)#.cuda()
y_train = torch.tensor(train_y, dtype=torch.long)#.cuda()
x_cv = torch.tensor(test_tokenized, dtype=torch.long)#.cuda()
y_cv = torch.tensor(test_y, dtype=torch.long)#.cuda()

x_valid = torch.tensor(valid_tokenized, dtype=torch.long)#.cuda()
y_valid = torch.tensor(valid_y, dtype=torch.long)#.cuda()


# Create Torch datasets
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_cv, y_cv)
valid = torch.utils.data.TensorDataset(x_valid, y_valid)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)


train_loss = []
valid_loss = []
print("Start training:")

for epoch in range(n_epochs):
    start_time = time.time()
    # Set model to train configuration
    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Predict/Forward Pass
        y_pred = model(x_batch)
        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    # Set model to validation configuration -Doesn't get trained here
    model.eval()
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_cv), len(le.classes_)))

    for i, (x_batch, y_batch) in enumerate(test_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(test_loader)
        # keep/store predictions
        val_preds[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=0).cpu().numpy()

        logits = y_pred.cpu().numpy()
        labels = y_batch.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, labels)



    # Check Accuracy
    val_accuracy = sum(val_preds.argmax(axis=1) == test_y) / len(test_y)
    train_loss.append(avg_loss)
    valid_loss.append(avg_val_loss)
    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))


torch.save(model,'bilstm_model')


def make_predictions(data_loader):
    predictions = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        y_pred = model(x_batch).detach()

        # keep/store predictions

        pred = F.softmax(y_pred, dim=0).cpu().numpy()

        pred = pred.argmax(axis=1)

       # pred = le.classes_[pred]

#        logits = y_pred.cpu().numpy()
#       labels = y_batch.to('cpu').numpy()

#        preds = torch.argmax(logits, dim=1)
        predictions.append(pred)

    return np.concatenate(predictions, axis=0)

#df_train['Prediction_x'] = make_predictions(train_loader)
#df_valid['Prediction_x'] = make_predictions(valid_loader)
df_test['Prediction_x'] = make_predictions(test_loader)

true_labels = torch.tensor(df_test['Label_x'].tolist())

#predictions = np.concatenate(predictions, axis=0)
predicted_labels = torch.tensor(df_test['Prediction_x'].tolist())

print(classification_report(true_labels, predicted_labels))

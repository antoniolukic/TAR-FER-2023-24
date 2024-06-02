import os
import json
import pandas as pd

def extract_reddit_source_data(directory):
    records = []

    for root, dirs, files in os.walk(directory):
        if 'source-tweet' in dirs:
            source_tweet_dir = os.path.join(root, 'source-tweet')
            for file in os.listdir(source_tweet_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(source_tweet_dir, file)
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        title = data['data']['children'][0]['data']['title']
                        records.append((file[:-5], title))

    df = pd.DataFrame(records, columns=['Key', 'Text'])
    return df

def extract_twitter_source_data(directory):
    records = []

    for root, dirs, files in os.walk(directory):
        if 'source-tweet' in dirs:
            source_tweet_dir = os.path.join(root, 'source-tweet')
            for file in os.listdir(source_tweet_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(source_tweet_dir, file)
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        tweet_id = str(data['id'])
                        tweet_text = data['text']
                        records.append((tweet_id, tweet_text))

    df = pd.DataFrame(records, columns=['Key', 'Text'])
    return df

def add_keys_from_json(df, json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        key_mapping = data["subtaskaenglish"]

    df['Label'] = df['Key'].map(key_mapping)
    label_mapping = {
        'support': 0,
        'deny': 1,
        'query': 2,
        'comment': 3
    }
    df['Label'] = df['Label'].map(label_mapping)
    return df

def get_source_data():
    directory_reddit = 'dataset/rumoureval-2019-training-data/reddit-training-data'
    directory_twitter = 'dataset/rumoureval-2019-training-data/twitter-english'

    df_r = extract_reddit_source_data(directory_reddit)
    df_r = add_keys_from_json(df_r, 'dataset/rumoureval-2019-training-data/train-key.json')
    df_t = extract_twitter_source_data(directory_twitter)
    df_t = add_keys_from_json(df_t, 'dataset/rumoureval-2019-training-data/train-key.json')
    df_r = df_r.dropna(subset=['Label'])  # there exists some NaN rows?!!
    df_t = df_t.dropna(subset=['Label'])  # there exists some NaN rows?!!
    
    df_r.to_csv('source_redit.csv', sep='\t', index=False)
    df_t.to_csv('source_twitter.csv', sep='\t', index=False)
    df_train = pd.concat([df_r, df_t], axis = 0)

    #df_train = add_keys_from_json(df_train, 'dataset/rumoureval-2019-training-data/train-key.json')
    #df_train = df_train.dropna(subset=['Label'])  # there exists some NaN rows?!!

    directory_valid = 'dataset/rumoureval-2019-training-data/reddit-dev-data'
    df_valid = extract_reddit_source_data(directory_valid)
    df_valid = add_keys_from_json(df_valid, 'dataset/rumoureval-2019-training-data/dev-key.json')
    df_valid = df_valid.dropna(subset=['Label'])

    df_valid.to_csv('source_valid.csv', sep='\t', index=False)

    return df_train, df_valid

def extract_reddit_reply_data(directory):
    records = []

    for root, dirs, files in os.walk(directory):
        if 'source-tweet' in dirs and 'replies' in dirs:
            source_tweet_dir = os.path.join(root, 'source-tweet')
            replies_dir = os.path.join(root, 'replies')
            source_files = os.listdir(source_tweet_dir)
            if source_files:
                source_file = source_files[0]
                source_key = source_file[:-5]
                for file in os.listdir(replies_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(replies_dir, file)
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            text = data.get('data', {}).get('body', None)
                            if text is not None:
                                records.append((source_key, file[:-5], text))

    df = pd.DataFrame(records, columns=['SourceKey', 'Key', 'Text'])
    return df

def extract_twitter_reply_data(directory):
    records = []

    for root, dirs, files in os.walk(directory):
        if 'source-tweet' in dirs and 'replies' in dirs:
            source_tweet_dir = os.path.join(root, 'source-tweet')
            replies_dir = os.path.join(root, 'replies')
            source_files = os.listdir(source_tweet_dir)
            if source_files:
                source_file = source_files[0] 
                source_key = source_file[:-5]
                for file in os.listdir(replies_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(replies_dir, file)
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            tweet_id = str(data['id'])
                            tweet_text = data['text']
                            records.append((source_key, tweet_id, tweet_text))

    df = pd.DataFrame(records, columns=['SourceKey', 'Key', 'Text'])
    return df

def get_reply_data():
    directory_reddit = 'dataset/rumoureval-2019-training-data/reddit-training-data'
    directory_twitter = 'dataset/rumoureval-2019-training-data/twitter-english'

    df_r = extract_reddit_reply_data(directory_reddit)
    df_r = add_keys_from_json(df_r, 'dataset/rumoureval-2019-training-data/train-key.json')
    df_t = extract_twitter_reply_data(directory_twitter)
    df_t = add_keys_from_json(df_t, 'dataset/rumoureval-2019-training-data/train-key.json')
    df_r = df_r.dropna(subset=['Label', 'SourceKey', 'Key'])
    df_t = df_t.dropna(subset=['Label', 'SourceKey', 'Key'])

    df_r.to_csv('reply_redit.csv', sep='\t', index=False)
    df_t.to_csv('reply_twitter.csv', sep='\t', index=False)
    df_train = pd.concat([df_r, df_t], axis = 0)
    
    #df_train = add_keys_from_json(df_train, 'dataset/rumoureval-2019-training-data/train-key.json')
    #df_train = df_train.dropna(subset=['Label', 'SourceKey', 'Key'])

    directory_valid = 'dataset/rumoureval-2019-training-data/reddit-dev-data'
    df_valid = extract_reddit_reply_data(directory_valid)
    df_valid = add_keys_from_json(df_valid, 'dataset/rumoureval-2019-training-data/dev-key.json')
    df_valid = df_valid.dropna(subset=['Label', 'SourceKey', 'Key'])

    df_valid.to_csv('reply_valid.csv', sep='\t', index=False)

    return df_train, df_valid

def get_test_data():
    directory_reddit = 'dataset/rumoureval-2019-test-data/reddit-test-data'
    directory_twitter = 'dataset/rumoureval-2019-test-data/twitter-en-test-data'

    df_r = extract_reddit_source_data(directory_reddit)
    df_r = add_keys_from_json(df_r, 'dataset/final-eval-key.json')
    df_t = extract_twitter_source_data(directory_twitter)
    df_t = add_keys_from_json(df_t, 'dataset/final-eval-key.json')
    df_r = df_r.dropna(subset=['Label'])  # there exists some NaN rows?!!
    df_t = df_t.dropna(subset=['Label'])  # there exists some NaN rows?!!
    
    df_r.to_csv('test_source_redit.csv', sep='\t', index=False)
    df_t.to_csv('test_source_twitter.csv', sep='\t', index=False)
    df_test_source = pd.concat([df_r, df_t], axis = 0)

    df_r = extract_reddit_reply_data(directory_reddit)
    df_r = add_keys_from_json(df_r, 'dataset/final-eval-key.json')
    df_t = extract_twitter_reply_data(directory_twitter)
    df_t = add_keys_from_json(df_t, 'dataset/final-eval-key.json')
    df_r = df_r.dropna(subset=['Label'])  # there exists some NaN rows?!!
    df_t = df_t.dropna(subset=['Label'])  # there exists some NaN rows?!!
    
    df_r.to_csv('test_reply_redit.csv', sep='\t', index=False)
    df_t.to_csv('test_reply_twitter.csv', sep='\t', index=False)
    df_test_reply = pd.concat([df_r, df_t], axis = 0)

    return df_test_source, df_test_reply


if __name__ == '__main__':
    get_source_data()
    get_reply_data()
    get_test_data()

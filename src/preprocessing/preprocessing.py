import pandas as pd

# Load Liar dataset.csv
liar_cols = ["id", "label", "statement"]
liar_path_test = "../../dataset/test.tsv"
liar_df_test = pd.read_csv(liar_path_test, sep='\t', header=None, usecols=[0, 1, 2], names=liar_cols)
liar_path_train = "../../dataset/train.tsv"
liar_df_train = pd.read_csv(liar_path_train, sep='\t', header=None, usecols=[0, 1, 2], names=liar_cols)
liar_path_valid = "../../dataset/valid.tsv"
liar_df_valid = pd.read_csv(liar_path_valid, sep='\t', header=None, usecols=[0, 1, 2], names=liar_cols)

# Load FakeNewsNet dataset.csv
# fnn_cols = ["id","news_url", "title", "tweet_id"]
fnn_cols = ["id", "statement", ]

fnn_path_gossip_fake = "../../dataset/gossipcop_fake.csv"
fnn_df_gossip_fake = pd.read_csv(fnn_path_gossip_fake, usecols=[0, 2], names=fnn_cols, header=0)
fnn_df_gossip_fake["label"] = "false"

fnn_path_gossip_real = "../../dataset/gossipcop_real.csv"
fnn_df_gossip_real = pd.read_csv(fnn_path_gossip_real, usecols=[0, 2], names=fnn_cols, header=0)
fnn_df_gossip_real["label"] = "true"

fnn_path_polit_fake = "../../dataset/politifact_fake.csv"
fnn_df_polit_fake = pd.read_csv(fnn_path_polit_fake, usecols=[0, 2], names=fnn_cols, header=0)
fnn_df_polit_fake["label"] = "false"

fnn_path_polit_real = "../../dataset/politifact_real.csv"
fnn_df_polit_real = pd.read_csv(fnn_path_polit_real, usecols=[0, 2], names=fnn_cols, header=0)
fnn_df_polit_real["label"] = "true"
combined_liar = pd.concat([liar_df_test, liar_df_train, liar_df_valid])
# Removing rows with missing values
combined_liar.dropna(inplace=True)
# Converting labels to numerical values
# I hate this! We lose so much information!
combined_liar['label'] = combined_liar['label'].map(
    {'pants-fire': 0, 'false': 0, 'barely-true': 0, 'half-true': 1, 'mostly-true': 1, 'true': 1})

combined_fnn = pd.concat([fnn_df_gossip_fake, fnn_df_gossip_real, fnn_df_polit_fake, fnn_df_polit_real])
# Removing rows with missing values

combined_fnn.dropna(inplace=True)
# Converting labels to numerical values
combined_fnn['label'] = combined_fnn['label'].map({'false': 0, 'true': 1})

combined_liar_fnn = pd.concat([combined_liar[["statement", "label"]], combined_fnn])
# print(combined_liar_fnn)
# combined_liar_fnn.to_csv('raw_dataset_all.csv', index=False)
# exit(0)
import string
import nltk

downloaded = True
if not downloaded:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords


# Define a function to preprocess text data
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation + '-–‘’“”'))

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)

    return text


# Preprocess the text data
combined_liar_fnn['statement'] = combined_liar_fnn['statement'].apply(preprocess_text)

import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Example dataset.csv of news articles with keywords
dataset = combined_liar_fnn
dataset = dataset[['statement', 'label']]

dataset_filepath = '../../dataset/dataset.csv'
dataset.to_csv(dataset_filepath, index=False)
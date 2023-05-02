# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:18:51 2023

@author: crist
"""

import pandas as pd

# Load Liar dataset
liar_cols = ["id", "label", "statement"]
liar_path_test = "D:/DataMining_FakeNews/dataset/test.tsv"
liar_df_test = pd.read_csv(liar_path_test, sep='\t', header=None, usecols=[0, 1, 2], names=liar_cols)
liar_path_train = "D:/DataMining_FakeNews/dataset/train.tsv"
liar_df_train = pd.read_csv(liar_path_train, sep='\t', header=None, usecols=[0, 1, 2], names=liar_cols)
liar_path_valid = "D:/DataMining_FakeNews/dataset/valid.tsv"
liar_df_valid = pd.read_csv(liar_path_valid, sep='\t', header=None, usecols=[0, 1, 2], names=liar_cols)

#%%
# Load FakeNewsNet dataset
#fnn_cols = ["id","news_url", "title", "tweet_id"]
fnn_cols = ["id","statement",]

fnn_path_gossip_fake = "D:/DataMining_FakeNews/dataset/gossipcop_fake.csv"
fnn_df_gossip_fake = pd.read_csv(fnn_path_gossip_fake, usecols=[0, 2], names=fnn_cols, header = 0)
fnn_df_gossip_fake["label"] = "false"

fnn_path_gossip_real = "D:/DataMining_FakeNews/dataset/gossipcop_real.csv"
fnn_df_gossip_real = pd.read_csv(fnn_path_gossip_real, usecols=[0,2], names=fnn_cols,  header = 0)
fnn_df_gossip_real["label"] = "true"

fnn_path_polit_fake = "D:/DataMining_FakeNews/dataset/politifact_fake.csv"
fnn_df_polit_fake = pd.read_csv(fnn_path_polit_fake, usecols=[0, 2], names=fnn_cols,  header = 0)
fnn_df_polit_fake["label"] = "false"

fnn_path_polit_real = "D:/DataMining_FakeNews/dataset/politifact_real.csv"
fnn_df_polit_real = pd.read_csv(fnn_path_polit_real, usecols=[0, 2], names=fnn_cols,  header = 0)
fnn_df_polit_real["label"] = "true"
combined_liar = pd.concat([liar_df_test, liar_df_train, liar_df_valid])
# Removing rows with missing values
combined_liar.dropna(inplace=True)
# Converting labels to numerical values
#I hate this! We lose so much information!
combined_liar['label'] = combined_liar['label'].map({'pants-fire': 0, 'false': 0, 'barely-true': 0, 'half-true': 1, 'mostly-true': 1, 'true': 1})

#%%
combined_fnn  = pd.concat([fnn_df_gossip_fake, fnn_df_gossip_real, fnn_df_polit_fake, fnn_df_polit_real])
# Removing rows with missing values
combined_fnn.dropna(inplace=True)
# Converting labels to numerical values
combined_fnn['label'] = combined_fnn['label'].map({'false': 0, 'true': 1})

#%%
# combined_liar_fnn =pd.concat([combined_liar[["id", "label", "statement"]], combined_fnn])
combined_liar_fnn =pd.concat([combined_liar[["statement","label"]], combined_fnn])

#%%
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define a function to preprocess text data
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation+'-–‘’“”' ))
    
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

#%% Apriori

import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Example dataset of news articles with keywords
dataset = combined_liar_fnn
dataset = dataset[['statement','label']]

onlyreal_l = dataset[dataset['label'] == 1]
onlyreal_s = onlyreal_l[['statement']]

# Step 1: Load sentences
sentences = onlyreal_s['statement'].tolist()
# Preprocessing steps here, e.g., tokenization, stopword removal, etc.

# Step 2: Convert sentences to transactions
transactions = []
for sentence in sentences:
    transaction = sentence.split()  # Split sentence into items (words)
    transactions.append(transaction)

# Step 3: Create transaction dataset
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
dfvfd
#%% Step 4: Apply Apriori algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.004, use_colnames=True)
#%%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Extract rules or perform other analyses as needed

#%%
onlyfake_l = dataset[dataset['label'] == 0]
onlyfake_n = onlyfake_l[['statement']]

# Step 1: Load sentences
sentences_n = onlyfake_n['statement'].tolist()
# Preprocessing steps here, e.g., tokenization, stopword removal, etc.

# Step 2: Convert sentences to transactions
transactions_n = []
for sentence_n in sentences_n:
    transaction_n = sentence_n.split()  # Split sentence into items (words)
    transactions_n.append(transaction_n)

# Step 3: Create transaction dataset
te_n = TransactionEncoder()
te_ary_n = te_n.fit_transform(transactions_n)
transaction_df_n = pd.DataFrame(te_ary_n, columns=te_n.columns_)

#%% Step 4: Apply Apriori algorithm
frequent_itemsets_n = apriori(transaction_df_n, min_support=0.004, use_colnames=True)
#%%
rules_n = association_rules(frequent_itemsets_n, metric="confidence", min_threshold=0.5)
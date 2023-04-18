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

# Step 4: Apply Apriori algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Extract rules or perform other analyses as needed

#%%
# Convert dataset to one-hot encoded format

# te = TransactionEncoder()
# te_ary = te.fit(dataset).transform(dataset)
# df = pd.DataFrame(te_ary, columns=te.columns_)

# # Run Apriori algorithm to find frequent itemsets
# frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# # Extract association rules from frequent itemsets
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# # Filter rules based on criteria, e.g., antecedent and consequent
# fake_news_rules = rules[ (rules['antecedents'].apply(lambda x: 'fake' in x))
#                   | (rules['consequents'].apply(lambda x: 'fake' in x))]

# # Print the resulting fake news rules
# print("Fake News Rules:")
# print(fake_news_rules)
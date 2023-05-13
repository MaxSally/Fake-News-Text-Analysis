# -*- coding: utf-8 -*-

import pandas as pd
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_colwidth', None)
dataset_filepath = '../../dataset/dataset.csv'
dataset = pd.read_csv(dataset_filepath)

run = 2 #0 = real, 1 = fake, 2 = all
detailed_print = 1

if run == 0:
    # % Get only real news
    onlyreal_l = dataset[dataset['label'] == 1]
    onlyreal_s = onlyreal_l[['statement']]

    # Step 1: Load sentences
    sentences = onlyreal_s['statement'].tolist()
    # Preprocessing steps here, e.g., tokenization, stopword removal, etc.

    # Step 2: Convert sentences to transactions
    transactions = []
    for sentence in sentences:
        transaction = str(sentence).split()  # Split sentence into items (words)
        transactions.append(transaction)

    # Step 3: Create transaction dataset.csv
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

    print(transaction_df.shape)

    # %% Step 4: Apply Apriori algorithm
    frequent_itemsets = apriori(transaction_df, min_support=0.004, use_colnames=True)
    # 0.004 ~ occur at least 100 (exacly 98.3) times out of a total of 24575 transactions

    print(frequent_itemsets.shape)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
    selected_columns = ['antecedents', 'consequents', 'support', 'confidence']

    rules.to_csv('real_news_rules.csv', index=False)
    print(rules.loc[:, selected_columns])
    antecedents_set = set(rules['antecedents'].tolist())
    consequents_set = set(rules['consequents'].tolist())
    combined_set = antecedents_set | consequents_set

    words = [list(word_set)[0] for word_set in combined_set]
    unique_words = set(words)
    print(unique_words)

    # Sort the rules based on support in descending order
    sorted_rules = rules.sort_values('confidence', ascending=False)

    # Print the top 10 rules with highest support
    print(sorted_rules.head(20))

elif run == 1:
    # % Get only fake news

    onlyfake_l = dataset[dataset['label'] == 0]
    onlyfake_n = onlyfake_l[['statement']]

    # Step 1: Load sentences
    sentences_n = onlyfake_n['statement'].tolist()
    # Preprocessing steps here, e.g., tokenization, stopword removal, etc.

    # Step 2: Convert sentences to transactions
    transactions_n = []
    for sentence_n in sentences_n:
        transaction_n = str(sentence_n).split()  # Split sentence into items (words)
        transactions_n.append(transaction_n)

    # Step 3: Create transaction dataset.csv
    te_n = TransactionEncoder()
    te_ary_n = te_n.fit_transform(transactions_n)
    transaction_df_n = pd.DataFrame(te_ary_n, columns=te_n.columns_)

    frequent_itemsets_n = apriori(transaction_df_n, min_support=0.004, use_colnames=True)

    print(frequent_itemsets_n.shape)

    rules_n = association_rules(frequent_itemsets_n, metric="confidence", min_threshold=0.9)
    selected_columns = ['antecedents', 'consequents', 'support', 'confidence']
    rules_n.to_csv('fake_news_rules.csv', index=False)
    print(rules_n.loc[:, selected_columns])

    antecedents_set_n = set(rules_n['antecedents'].tolist())
    consequents_set_n = set(rules_n['consequents'].tolist())
    combined_set_n = antecedents_set_n | consequents_set_n

    words_n = [list(word_set_n)[0] for word_set_n in combined_set_n]
    unique_words_n = set(words_n)
    print(unique_words_n)

    # Sort the rules based on support in descending order
    sorted_rules = rules_n.sort_values('confidence', ascending=False)

    # Print the top 10 rules with highest support
    print(sorted_rules.head(20))


elif run == 2:
    # %% All data
    All = dataset

    All_s = All[['statement']]

    # Step 1: Load sentences
    sentences_All = All_s['statement'].tolist()
    # Preprocessing steps here, e.g., tokenization, stopword removal, etc.

    # Step 2: Convert sentences to transactions
    transactions_All = []
    for sentence in sentences_All:
        transaction_All = str(sentence).split()  # Split sentence into items (words)
        transactions_All.append(transaction_All)

    # Step 3: Create transaction dataset.csv
    te_All = TransactionEncoder()
    te_ary_All = te_All.fit_transform(transactions_All)
    transaction_df_All = pd.DataFrame(te_ary_All, columns=te_All.columns_)

    # %% Step 4: Apply Apriori algorithm
    frequent_itemsets_All = apriori(transaction_df_All, min_support=0.004, use_colnames=True)

    rules_All = association_rules(frequent_itemsets_All, metric="confidence", min_threshold=0.9)

    print(frequent_itemsets_All.shape)

    selected_columns_All = ['antecedents', 'consequents', 'support', 'confidence']
    rules_All.to_csv('entire_dataset_rules.csv', index=False)
    print(rules_All.loc[:, selected_columns_All])

    antecedents_set_All = set(rules_All['antecedents'].tolist())
    consequents_set_All = set(rules_All['consequents'].tolist())
    combined_set_All = antecedents_set_All | consequents_set_All

    words_All = [list(word_set_All)[0] for word_set_All in combined_set_All]
    unique_words_All = set(words_All)
    print(unique_words_All)

    # Sort the rules based on support in descending order
    sorted_rules = rules_All.sort_values('confidence', ascending=False)

    # Print the top 10 rules with highest support
    print(sorted_rules.head(20))

    # Set max_rows to display all the results
    pd.options.display.max_rows = len(rules_All)

    # Print the association rules dataframe
    print(rules_All)
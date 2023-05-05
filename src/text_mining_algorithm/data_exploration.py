import csv

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sia = SentimentIntensityAnalyzer()
superlatives = ["fastest", "quickest", "tallest", "longest", "shortest", "oldest", "freshest", "heaviest", "lightest",
                "brightest", "darkest", "clearest", "foggiest", "hottest", "coldest", "busiest", "loudest", "quietest",
                "smallest", "biggest", "most expensive", "least expensive", "most popular", "least popular",
                "most talented", "least talented", "most successful", "least successful", "most beautiful",
                "least beautiful", "most delicious", "least delicious", "most exciting", "least exciting",
                "most important", "least important", "most interesting", "least interesting", "most valuable",
                "least valuable", "most intelligent", "least intelligent", "most useful", "least useful",
                "most versatile", "least versatile", "most innovative", "least innovative", "most inspiring",
                "least inspiring", "most friendly", "least friendly", "most reliable", "least reliable",
                "most efficient", "least efficient", "most creative", "least creative", "most comfortable",
                "least comfortable", "most exciting", "least exciting", "most entertaining", "least entertaining",
                "most enjoyable", "least enjoyable", "most challenging", "least challenging", "most rewarding",
                "least rewarding", "most memorable", "least memorable", "most impressive", "least impressive",
                "most convenient", "least convenient", "most addictive", "least addictive", "most attractive",
                "least attractive", "most durable", "least durable", "most reliable", "least reliable",
                "most satisfying", "least satisfying", "most stunning", "least stunning", "most amazing",
                "least amazing", "most fascinating", "least fascinating", "most relaxing", "least relaxing",
                "most refreshing", "least refreshing", "most surprising", "least surprising", "most significant",
                "least significant", "most fulfilling", "least fulfilling", "most impressive", "least impressive",
                "most incredible", "least incredible", "most appealing", "least appealing", "most inviting",
                "least inviting", "most alluring", "least alluring", "most beneficial", "least beneficial",
                "most helpful", "least helpful", "most satisfying", "least satisfying", "most delicious",
                "least delicious", "most nutritious", "least nutritious", "most delightful", "least delightful",
                "most relaxing", "least relaxing", "most comfortable", "least comfortable", "most innovative",
                "least innovative", "most convenient", "least convenient", "most user-friendly", "least user-friendly"]

comparatives = ["faster", "quicker", "taller", "shorter", "longer", "older", "younger", "stronger", "weaker", "heavier",
                "lighter", "wider", "narrower", "higher", "lower", "brighter", "darker", "smoother", "rougher",
                "thinner", "fatter", "sharper", "duller", "cleaner", "dirtier", "safer", "riskier", "cooler", "warmer",
                "fresher", "staler", "harder", "softer", "busier", "calmer", "quieter", "louder", "easier", "harder",
                "higher", "lower", "cheaper", "more expensive", "more important", "less important", "more interesting",
                "less interesting", "more popular", "less popular", "more successful", "less successful",
                "more beautiful", "less beautiful", "more confident", "less confident", "more effective",
                "less effective", "more efficient", "less efficient", "more skilled", "less skilled", "more reliable",
                "less reliable", "more powerful", "less powerful", "more intelligent", "less intelligent",
                "more creative", "less creative", "more adventurous", "less adventurous", "more romantic",
                "less romantic", "more polite", "less polite", "more patient", "less patient", "more responsible",
                "less responsible", "more flexible", "less flexible", "more organized", "less organized",
                "more experienced", "less experienced", "more knowledgeable", "less knowledgeable", "more ambitious",
                "less ambitious", "more compassionate", "less compassionate", "more honest", "less honest",
                "more humble", "less humble", "more loyal", "less loyal", "more optimistic", "less optimistic",
                "more punctual", "less punctual", "more respectful", "less respectful"]

extreme_words = ['outrageous', 'unbelievable', 'astonishing', 'unprecedented', 'unimaginable', 'unthinkable',
                 'inconceivable', 'mind-boggling', 'mind-blowing', 'jaw-dropping', 'eye-popping', 'staggering',
                 'mind-numbing', 'breathtaking', 'astounding', 'startling', 'shocking', 'amazing', 'incredible',
                 'fantastic', 'phenomenal', 'extraordinary', 'remarkable', 'unreal', 'miraculous', 'spectacular',
                 'marvelous', 'awe-inspiring', 'exquisite', 'wondrous', 'supreme', 'ultimate', 'unsurpassed',
                 'incomparable', 'peerless', 'matchless', 'unequaled', 'unmatched', 'unrivaled', 'unbeatable',
                 'undefeatable', 'unassailable', 'invincible', 'insuperable', 'impossible', 'unstoppable',
                 'uncontrollable', 'irresistible', 'overwhelming', 'incredible', 'phenomenal', 'tremendous', 'enormous',
                 'huge', 'gigantic', 'colossal', 'massive', 'monstrous', 'gargantuan', 'titanic', 'astronomical',
                 'unbelievably', 'astonishingly', 'unprecedentedly', 'unimaginably', 'unthinkably', 'inconceivably',
                 'mind-bogglingly', 'mind-blowingly', 'jaw-droppingly', 'eye-poppingly', 'staggeringly',
                 'mind-numbingly', 'breathtakingly', 'astoundingly', 'startlingly', 'shockingly', 'amazingly',
                 'incredibly', 'fantastically', 'phenomenally', 'extraordinarily', 'remarkably', 'unusually',
                 'miraculously', 'spectacularly', 'marvelously', 'awe-inspiringly', 'exquisitely', 'wondrously',
                 'supremely', 'ultimately', 'unsurpassedly', 'incomparably', 'peerlessly', 'matchlessly', 'unequaledly',
                 'unmatchedly', 'unrivaledly', 'unbeatably', 'undefeatably', 'unassailably', 'invincibly',
                 'insuperably', 'impossibly', 'unstoppably', 'uncontrollably', 'irresistibly', 'overwhelmingly',
                 'incredibly', 'phenomenally', 'tremendously', 'enormously', 'hugely', 'gigantically', 'colossally',
                 'massively', 'monstrously', 'gargantuanly', 'titanically', 'astronomically']

import re


def has_pronoun(text):
    pronoun_regex = r"\b(I|me|my|mine|myself|you|your|yours|yourself|he|him|his|himself|she|her|hers|herself|it|its|itself|we|us|our|ours|ourselves|you|your|yours|yourselves|they|them|their|theirs|themselves)\b"
    return bool(re.search(pronoun_regex, text, re.IGNORECASE))


def has_pronoun_first_seconod(text):
    pronoun_regex = r"\b(I|me|my|mine|myself|you|your|yours|yourself|we|us|our|ours|ourselves|you|your|yours|yourselves)\b"
    return bool(re.search(pronoun_regex, text, re.IGNORECASE))


def check_sensational_language(text):
    sensational_words = ['shocking', 'outrageous', 'explosive', 'heartbreaking', 'devastating', 'tragic', 'bombshell',
                         'stunning', 'mind-blowing', 'jaw-dropping', 'unbelievable', 'mind-boggling', 'game-changing',
                         'earth-shattering', 'breathtaking', 'massive', 'unprecedented', 'exclusive', 'blockbuster',
                         'breakthrough', 'life-changing', 'extraordinary', 'astonishing', 'explosive revelation',
                         'bombshell report', 'shocking new details', 'breaking news', 'never-before-seen',
                         'shocking twist', 'shocking allegation', 'dramatic turn', 'shocking discovery',
                         'stunning revelation', 'exclusive interview', 'blockbuster report', 'breakthrough discovery',
                         'life-altering', 'stunning discovery', 'huge development', 'game-changer', 'revolutionary',
                         'historic', 'epic', 'incredible', 'incredible discovery', 'shocking video', 'shocking photos',
                         'explosive video', 'explosive photos', 'unbelievable video', 'unbelievable photos',
                         'jaw-dropping video', 'jaw-dropping photos', 'mind-blowing video', 'mind-blowing photos',
                         'earth-shattering discovery', 'never-before-seen footage', 'tragic twist',
                         'devastating effects', 'heartbreaking footage', 'mind-boggling discovery',
                         'jaw-dropping discovery', 'unbelievable discovery', 'game-changing discovery',
                         'explosive allegations', 'bombshell announcement', 'breathtaking discovery',
                         'exclusive access', 'blockbuster revelation', 'massive leak', 'unprecedented discovery',
                         'historic announcement', 'epic discovery', 'incredible footage', 'incredible claims',
                         'unbelievable claims', 'jaw-dropping claims', 'shocking claims', 'mind-blowing claims',
                         'earth-shattering claims', 'never-before-seen evidence', 'explosive evidence',
                         'unbelievable evidence', 'jaw-dropping evidence', 'mind-blowing evidence',
                         'game-changing evidence', 'earth-shattering evidence', 'massive cover-up',
                         'blockbuster exposé', 'explosive exposé', 'shocking exposé', 'jaw-dropping exposé',
                         'unbelievable exposé', 'mind-blowing exposé', 'earth-shattering exposé',
                         'unprecedented exposé', 'exclusive exposé', 'never-before-seen exposé', 'massive scandal',
                         'blockbuster scandal', 'explosive scandal', 'shocking scandal', 'jaw-dropping scandal',
                         'unbelievable scandal', 'mind-blowing scandal', 'earth-shattering scandal',
                         'unprecedented scandal', 'exclusive scandal', 'never-before-seen scandal']
    for word in sensational_words:
        if word in text.lower():
            return True
    return False


def sentiment_analysis(text):
    # Initialize the sentiment analyzer

    # Get the sentiment score for the text
    sentiment_score = sia.polarity_scores(text)

    # Print the sentiment score
    return sentiment_score

score_list_real = []
score_list_fake = []

filepath = '../../dataset/raw_dataset_all.csv'
with open(filepath, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    line_count = 0
    cnt1_superlatives = 0
    cnt0_superlatives = 0
    cnt1_comparatives = 0
    cnt0_comparatives = 0
    cnt1_exaggerating = 0
    cnt0_exaggerating = 0
    cnt_row = 0
    cnt_true = 0
    cnt_false = 0
    cnt1_pronoun = 0
    cnt0_pronoun = 0
    cnt1_sensational = 0
    cnt0_sensational = 0
    cnt1_pronoun_first = 0
    cnt0_pronoun_first = 0
    cnt_sentiment = [0, 0, 0, 0]
    for row in reader:
        cnt_row += 1
        line_count += 1
        if int(row[1]) == 1:
            cnt_true += 1
        else:
            cnt_false += 1
        statement = str(row[0]).lower()
        # for superlative in superlatives:
        #     if superlative in statement:
        #         # print(f"Line {line_count} contains a superlative, and label is {row[1]}")
        #         if int(row[1]) == 1:
        #             cnt1_superlatives += 1
        #         else:
        #             cnt0_superlatives += 1
        #
        # for comparative in comparatives:
        #     if comparative in statement:
        #         # print(f"Line {line_count} contains a superlative, and label is {row[1]}")
        #         if int(row[1]) == 1:
        #             cnt1_comparatives += 1
        #         else:
        #             cnt0_comparatives += 1
        #
        # for exaggerating_word in extreme_words:
        #     if exaggerating_word in statement:
        #         # print(f"Line {line_count} contains a superlative, and label is {row[1]}")
        #         if int(row[1]) == 1:
        #             cnt1_exaggerating += 1
        #         else:
        #             cnt0_exaggerating += 1
        #
        # if has_pronoun(statement):
        #     if int(row[1]) == 1:
        #         cnt1_pronoun += 1
        #     else:
        #         cnt0_pronoun += 1
        #
        # if has_pronoun_first_seconod(statement):
        #     if int(row[1]) == 1:
        #         cnt1_pronoun_first += 1
        #     else:
        #         cnt0_pronoun_first += 1
        #
        # if check_sensational_language(statement):
        #     if int(row[1]) == 1:
        #         cnt1_sensational += 1
        #     else:
        #         cnt0_sensational += 1

        score = sentiment_analysis(statement)
        if int(row[1]) == 1:
            score_list_real.append(score['compound'])
        else:
            score_list_fake.append(score['compound'])
        if score['compound'] > 0.5:
            if int(row[1]) == 1:
                cnt_sentiment[0] += 1
            else:
                cnt_sentiment[1] += 1
        elif score['compound'] < -0.5:
            if int(row[1]) == 1:
                cnt_sentiment[2] += 1
            else:
                cnt_sentiment[3] += 1

    ratio = 1.0 * cnt_false / cnt_true
    # print('Superlatives ', cnt1_superlatives, cnt1_superlatives * ratio, cnt0_superlatives,
    #       100.0 * cnt0_superlatives / (cnt1_superlatives * ratio + cnt0_superlatives))
    # print('Comparatives ', cnt1_comparatives, cnt1_comparatives * ratio, cnt0_comparatives,
    #       100.0 * cnt0_comparatives / (cnt1_comparatives * ratio + cnt0_comparatives))
    # print('Exaggerating ', cnt1_exaggerating, cnt1_exaggerating * ratio, cnt0_exaggerating,
    #       100.0 * cnt0_exaggerating / (cnt1_exaggerating * ratio + cnt0_exaggerating))
    # print('Sensational ', cnt1_sensational, cnt1_sensational * ratio, cnt0_sensational,
    #       100.0 * cnt0_sensational / (cnt1_sensational * ratio + cnt0_sensational))
    # print('Pronoun ', cnt1_pronoun, cnt1_pronoun * ratio, cnt0_pronoun,
    #       100.0 * cnt0_pronoun / (cnt1_pronoun * ratio + cnt0_pronoun))
    # print('First and second Pronoun ', cnt1_pronoun_first, cnt1_pronoun_first * ratio, cnt0_pronoun_first,
    #       100.0 * cnt0_pronoun_first / (cnt1_pronoun_first * ratio + cnt0_pronoun_first))
    print('cnt ', cnt_true, cnt_false)
    print('Sentiment ', cnt_sentiment)
    print('Sentiment_weighted ', cnt_sentiment[0] * ratio, cnt_sentiment[1], cnt_sentiment[2] * ratio, cnt_sentiment[3])
    print(cnt_row)
    print(ratio)

    import seaborn as sns

    sns.set_style("whitegrid")

    # Plot histogram of sentiment scores
    plt.hist(score_list_real, bins=10, range=(-1, 1), color='blue', alpha=0.5)
    plt.title('Sentiment Real News Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(score_list_fake, bins=10, range=(-1, 1), color='blue', alpha=0.5)
    plt.title('Sentiment False News Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

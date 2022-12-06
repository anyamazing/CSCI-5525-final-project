# Code adapted from work by Nicha Ruchirawat
# https://nicharuc.github.io/topic_modeling/
import nltk
import pandas as pd

###############################
# Bigrams
###############################
def bigram_filter(bigram, stop_word_list):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in stop_word_list or bigram[1] in stop_word_list:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    return True

def get_bigrams(data_samples, occur_rate, pmi_threshold):
    ds = pd.DataFrame(data_samples)
    ds.columns = ['review']

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents([r.split() for r in ds.review])

    # Filter only those bigrams that occur at least occur_rate times
    finder.apply_freq_filter(occur_rate)
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)

    bigram_pmi = pd.DataFrame(bigram_scores)
    bigram_pmi.columns = ['bigram', 'pmi']
    bigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

    filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram: bigram_filter(bigram['bigram']) and bigram.pmi > pmi_threshold, axis = 1)][:500]

    bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
    return bigrams

###############################
# TRIGRAMS
###############################
def trigram_filter(trigram, stop_word_list):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if trigram[0] in stop_word_list or trigram[-1] in stop_word_list or trigram[1] in stop_word_list:
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    return True

def get_trigrams(data_samples, occur_rate, pmi_threshold):
    ds = pd.DataFrame(data_samples)
    ds.columns = ['review']
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.TrigramCollocationFinder.from_documents([r.split() for r in ds.review])

    # Filter only those that occur at least occur_rate times
    finder.apply_freq_filter(occur_rate)
    trigram_scores = finder.score_ngrams(trigram_measures.pmi)

    trigram_pmi = pd.DataFrame(trigram_scores)
    trigram_pmi.columns = ['trigram', 'pmi']
    trigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

    filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: trigram_filter(trigram['trigram']) and trigram.pmi > 17, axis = 1)][:500]
    trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]
    return trigrams

###############################
# NGRAMS
###############################
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x

def replace_n_grams(data_samples):
    ds = pd.DataFrame(data_samples)
    ds.columns = ['review']
    reviews_w_ngrams = ds.copy()
    reviews_w_ngrams.review = reviews_w_ngrams.review.map(lambda x: replace_ngram(x))
    _data_samples = reviews_w_ngrams.values.tolist()
    data_samples = [item for sublist in _data_samples for item in sublist]
    return data_samples

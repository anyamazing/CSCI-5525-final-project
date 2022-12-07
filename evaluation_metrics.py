import nltk
from nltk import word_tokenize
import math
from itertools import combinations
from tqdm import tqdm

def get_word_freqs(data_samples):
    freq = {}
    for sample in tqdm(data_samples,leave=True):
        for word in list(set(word_tokenize(sample))):
            if word in list(freq.keys()):
                freq[word] += 1
            else:
                freq[word] = 1
    freq.update((x, float(y)/len(data_samples)) for x,y in freq.items())

    return freq

def get_bi_freqs(data_samples):
    bi_freq = {}
    for sample in tqdm(data_samples,leave=True):
        combos = combinations(list(set(word_tokenize(sample))),2)
        for combo in combos:
            if combo in list(bi_freq.keys()):
                bi_freq[combo] += 1
            else:
                bi_freq[combo] = 1
    bi_freq.update((x, float(y)/len(data_samples)) for x,y in bi_freq.items())

    return bi_freq

def u_mass(features, freq, bi_freq, num_docs):
    pairs = combinations(features,2)
    total = 0
    eta = 1/num_docs
    for pair in pairs:
        try:
            num = bi_freq[pair]
        except:
            num = 0
        prob = math.log((num + eta)/freq[pair[1]])
        total += prob
    return total

def avg_umass(model, feature_names, n_top_words, freq, bi_freq, num_docs):
    coherences = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words-1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        coherences.append(u_mass(top_features, freq, bi_freq, num_docs))
    return coherences, sum(coherences)/len(coherences)

def jaccard(model, feature_names, n_top_words):
    similarities = []
    x = None
    y = None
    for i, primary_topic in enumerate(model.components_):
        t_similarities = []
        primary_features_ind = primary_topic.argsort()[: -n_top_words - 1 : -1]
        primary_features = [feature_names[j] for j in primary_features_ind]
        primary_features = set(primary_features)
        for k, topic in enumerate(model.components_):
            if i == k:
                pass
            else:
                features_ind = topic.argsort()[: -n_top_words - 1 : -1]
                features = [feature_names[j] for j in features_ind]
                features = set(features)

                intersection = len(list(primary_features.intersection(features)))
                union = (len(list(primary_features)) + len(list(features))) - intersection
                jac = float(intersection)/union
                t_similarities.append(jac)

        similarities.append(t_similarities)

    return similarities, x, y

def print_jaccard(similarities):
    print("Jaccard similarity coefficient between topics:")
    for i, _similarities in enumerate(similarities):
        print()
        for j, val in enumerate(_similarities):
            if j<i:
                pass
            else:
                print(str(i+1)+" and "+str(j+2)+": ", val)
    return

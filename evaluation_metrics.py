import nltk
import math
from itertools import combinations

def get_word_freqs(data_samples):
    freq = nltk.FreqDist()
    for sample in data_samples:
        for word in word_tokenize(sample):
            freq[word] += 1

    return freq

def get_bigram_freqs(data_samples):
    bi_freq = nltk.FreqDist()
    for sample in data_samples:
        bigrams = nltk.bigrams(word_tokenize(sample))
        for gram in bigrams:
            bi_freq[gram] += 1

    return bi_freq

def u_mass(features, freq, bi_freq, num_docs):
    pairs = combinations(features,2)
    total = 0
    eta = 1/num_docs
    for pair in pairs:
        prob = math.log((bi_freq.freq(pair) + eta)/freq.freq(pair[1]))
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
    return similarities

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

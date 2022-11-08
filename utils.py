from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def lowercasing(data_samples):
    """convert samples to lowercase"""
    for idx, sample in tqdm(enumerate(data_samples)):
        data_samples[idx] = sample.lower()
    return data_samples


def punctuation_removal(data_samples):
    # non-exhaustive; not sure if we want to treat punctuation as significant
    # doesn't remove punctuation from inside words
    for i, sample in tqdm(enumerate(data_samples)):
        _sample = sample.split()
        for j, word in enumerate(_sample):
            _sample[j] = word.strip(" .!?@#&():;,'\/\\")
        sample = " ".join(_sample)
        data_samples[i] = sample
    return data_samples

# credit to Selva Prabhakaran
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(data_samples):
    wnl = WordNetLemmatizer()
    for i, sample in tqdm(enumerate(data_samples)):
        _sample = sample.split()
        for j, word in enumerate(_sample):
            tag = get_wordnet_pos(word)
            _sample[j] = wnl.lemmatize(word, tag)
        data_samples[i] = " ".join(_sample)
    return data_samples

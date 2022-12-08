from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import sent_tokenize
from tqdm import tqdm
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


def lowercase(data_samples):
    """convert samples to lowercase"""
    for idx, sample in tqdm(enumerate(data_samples), leave=True):
        data_samples[idx] = sample.lower()
    return data_samples


def remove_punctuation(data_samples):
    """currently limited, does not remove punctuation from inside words"""
    for i, sample in tqdm(enumerate(data_samples), leave=True):
        _sample = sample.split()
        for j, word in enumerate(_sample):
            _sample[j] = word.strip(" .!?@#&():;,'\/\\")
        sample = " ".join(_sample)
        data_samples[i] = sample
    return data_samples


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    # credit to Selva Prabhakaran
    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(data_samples):
    wnl = WordNetLemmatizer()
    for i, sample in tqdm(enumerate(data_samples), leave=True):
        _sample = sample.split()
        for j, word in enumerate(_sample):
            tag = get_wordnet_pos(word)
            _sample[j] = wnl.lemmatize(word, tag)
        data_samples[i] = " ".join(_sample)
    return data_samples


PREPROCESSING_STEPS = [lowercase, remove_punctuation, lemmatize]


def preprocess_samples(samples):
    """processes all of the samples"""
    samples = samples[:]
    for step in PREPROCESSING_STEPS:
        print(f"Applying {step.__name__}")
        step(samples)
    return samples


def preprocess_sample(sample, get_raw=False):
    raw_sentences = sent_tokenize(sample)
    processed_sentences = raw_sentences[:]
    for step in PREPROCESSING_STEPS:
        print(f"Applying {step.__name__}")
        step(processed_sentences)
    return processed_sentences if not get_raw else (processed_sentences, raw_sentences)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import sent_tokenize
from tqdm import tqdm
import nltk
import re
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


def lowercase(data_samples):
    for idx, sample in tqdm(enumerate(data_samples), leave=True):
        data_samples[idx] = sample.lower()
    return data_samples

def remove_punctuation(data_samples):
  for i, sample in tqdm(enumerate(data_samples), leave=True):
    _sample = sample.split()
    for j, word in enumerate(_sample):
      _sample[j] = word.strip(" 0123456789\*\$\.!\?@#&\(\):;,'\/\\\"")
    sample = " ".join(_sample)
    data_samples[i] = sample
  return data_samples

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(data_samples):
  wnl = WordNetLemmatizer()
  for i, sample in tqdm(enumerate(data_samples), leave=True):
      _sample = sample.split()
      for j, word in enumerate(_sample):
        tag = get_wordnet_pos(word)
        if (tag == 'r') or (tag == 'a') or (tag == 'v'):
          _sample[j] = ""
        else:
          _sample[j] = wnl.lemmatize(word,tag)
      data_samples[i] = ' '.join(_sample)
      data_samples[i] = re.sub(' +', ' ', data_samples[i])

  return data_samples

PREPROCESSING_STEPS = [lowercase, remove_punctuation, lemmatize]


def preprocess_samples(samples):
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

import pandas as pd
from gradio.components import Textbox, HighlightedText, JSON
import gradio as gr
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import os
import pickle


def lowercasing(lda_samples):
    for idx, sample in tqdm(enumerate(lda_samples)):
        lda_samples[idx] = sample.lower()
    return lda_samples


def punctuation_removal(lda_samples):
    # non-exhaustive; not sure if we want to treat punctuation as significant
    # doesn't remove punctuation from inside words
    for i, sample in tqdm(enumerate(lda_samples)):
        _sample = sample.split()
        for j, word in enumerate(_sample):
            _sample[j] = word.strip(" .!?@#&():;,'\/\\")
        sample = " ".join(_sample)
        lda_samples[i] = sample
    return lda_samples


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(lda_samples):
    wnl = WordNetLemmatizer()
    for i, sample in tqdm(enumerate(lda_samples)):
        _sample = sample.split()
        for j, word in enumerate(_sample):
            tag = get_wordnet_pos(word)
            _sample[j] = wnl.lemmatize(word, tag)
        lda_samples[i] = " ".join(_sample)
    return lda_samples


def predict(text):
    raw_sentences = sent_tokenize(text)

    processed_sentences = raw_sentences[:]
    processed_sentences = lowercasing(processed_sentences)
    processed_sentences = punctuation_removal(processed_sentences)
    processed_sentences = lemmatize(processed_sentences)

    res = []
    present_topics = set()
    for raw, processed in zip(raw_sentences, processed_sentences):
        vs = analyzer.polarity_scores(raw)
        probs = lda.transform(tf_vectorizer.transform([processed]))[0]
        topic = probs.argmax()

        res.append((raw, f"Topic {topic+1} ({round(vs['compound'],2)})"))
        present_topics.add(topic)

    topics = {str(i + 1): ", ".join(topic_words[i]) for i in sorted(list(present_topics))}
    return [res, topics]


json_files = [pos_json for pos_json in os.listdir(".") if pos_json.endswith('.json')]

dfs = []
for f in json_files:
    dfs.append(pd.read_json(path_or_buf=f, lines=True))

df = pd.concat(dfs)


n_features = 1000
n_components = 10

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

with open("vectorizer.pkl", "rb") as f:
    tf_vectorizer = pickle.load(f)

product_id = 'B009MA34NY'
lda_samples = list(filter(lambda x: isinstance(x, str), df[df['asin'] == product_id]['reviewText']))
lda_samples = lowercasing(lda_samples)
lda_samples = punctuation_removal(lda_samples)
lda_samples = lemmatize(lda_samples)
documents = tf_vectorizer.transform(lda_samples)


lda = LatentDirichletAllocation(
    n_components=n_components,
    max_iter=5,
    learning_offset=50.0,
    random_state=0,
)
lda.fit(documents)

tf_feature_names = tf_vectorizer.get_feature_names_out()

raw_reviews = list(filter(lambda x: isinstance(x, str), df[df['asin'] == product_id]['reviewText']))
raw_sentences = sent_tokenize(raw_reviews[337])

processed_sentences = raw_sentences[:]
processed_sentences = lowercasing(processed_sentences)
processed_sentences = punctuation_removal(processed_sentences)
processed_sentences = lemmatize(processed_sentences)

feature_names = tf_vectorizer.get_feature_names_out()
topic_words = []
for topic in lda.components_:
    top_features_ind = topic.argsort()[: -10 - 1: -1]
    topic_words.append([feature_names[i] for i in top_features_ind])

analyzer = SentimentIntensityAnalyzer()


sentiment_vals = np.linspace(-1.0, 1.0, num=201)
color_map = {}
colors = {1: "red", 2: "orange", 3: "lime", 4: "pink", 5: "brown", 6: "green", 7: "purple", 8: "blue", 9: "cyan", 10: "yellow"}
for i, color in colors.items():
    color_map.update({f"Topic {i} ({round(val,2)})": color for val in sentiment_vals})

gr.Interface(fn=predict,
             inputs=Textbox(placeholder="Enter review here...", lines=5),
             outputs=[HighlightedText().style(color_map=color_map), JSON()],
             examples=[
                 ["Good indoor training shoes for running on treadmill, doing lunges and regular exercises at the gym. These are very flexible, light weight and comfortable. Grip is okay - sticky rubber is used only at the edges of heel and toe areas so I slipped a little when I worked on cable machines, resistance band, etc. on un-carpeted floor.  I would emphasize that if you do lifting as a part of your everyday routine workout I would not recommend them because mine (cushion) lasted only for six months and this is the reason I gave three stars. Other than that, I liked them!"],
                 ["I've had these shoes for about a week now and have so far enjoyed using them. Considering the fact that I have wide feet, the shoes are slightly tight. However, it doesn't feel uncomfortable nor does it bothers me as I use them throughout my workouts. I know some people personally like when the shoes are a bit tighter or a bit looser so it's all in personal preference."],
                 ["The picture makes the shoe look like it has a \"boxier\" toe rather than the \"pointier\" toe that it actually has. I have wider feet and generally need to buy a size or half size longer to get a comfortable width (in any brand of shoe). I was shooting for a rounder, broader toe design which is more comfortable for me, and I feel that the pictures of this shoe didn't accurately depict what I received, in that one detail. Otherwise, \"the shoe fits\" So I am wearing it."]
             ],
             ) \
    .launch(share=True)

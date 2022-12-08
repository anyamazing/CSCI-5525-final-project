import textblob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()

from data import generate_dataframe

df = generate_dataframe(["Digital_Music_5.json"])

# get all ratings and review text
# normalize to 0 to 1
ratings = df["overall"].values
ratings = (ratings - 1) / 4

reviews = [str(x) for x in df["reviewText"].tolist()]

# get the sentiment of each review with various tools

# textblob, normalized to be between 0 and 1
textblob_sentiments = [
    (textblob.TextBlob(review).sentiment.polarity + 1) / 2 for review in reviews
]

# vader, normalized to be between 0 and 1
vader_sentiments = [
    0.5 * (1 + analyzer.polarity_scores(review)["compound"]) for review in reviews
]

# calculate the correlation between the ratings and the sentiments
from scipy.stats import pearsonr

print("TextBlob: ", pearsonr(ratings, textblob_sentiments))
print("Vader: ", pearsonr(ratings, vader_sentiments))

# plot overlapping histograms of the ratings and the sentiments
import matplotlib.pyplot as plt

"""
# show vader distribution per rating, overlapping normalized histograms
for rating in range(1, 6):
    plt.hist(
        [x for x, y in zip(vader_sentiments, ratings) if y == rating],
        alpha=0.5,
        label=rating,
        density=True,
    )
plt.legend(loc="upper right")
plt.show()

# show textblob distribution per rating, overlapping normalized histograms
for rating in range(1, 6):
    plt.hist(
        [x for x, y in zip(textblob_sentiments, ratings) if y == rating],
        alpha=0.5,
        label=rating,
        density=True,
    )
plt.legend(loc="upper right")
plt.show()
"""

# put them side by side
# label the plots
colors = ["red", "orange", "yellow", "green", "blue"]

fig, (ax1, ax2) = plt.subplots(1, 2)
for rating in range(1, 6):
    ax1.hist(
        [x for x, y in zip(vader_sentiments, ratings) if y == rating],
        alpha=0.3,
        label=rating,
        bins=20,
        density=True,
        color=colors[rating - 1],
    )
ax1.set_title("Vader")
ax1.legend(loc="upper right")
for rating in range(1, 6):
    ax2.hist(
        [x for x, y in zip(textblob_sentiments, ratings) if y == rating],
        alpha=0.3,
        label=rating,
        bins=20,
        density=True,
        color=colors[rating - 1],
    )
ax2.set_title("TextBlob")
ax2.legend(loc="upper right")
plt.show()

import os
from preprocessing import preprocess_samples
from utils import get_product_reviews
from data import generate_dataframe


# read each file in the summary output directory
run_name = "music"
input_file = "Digital_Music_5.json"
output_dir = f"qt/outputs/{run_name}/"

df = generate_dataframe([input_file])

raw_corpus_samples = list(filter(lambda x: isinstance(x, str), df["reviewText"]))
corpus_samples = preprocess_samples(raw_corpus_samples)


for file in os.listdir(output_dir):
    # read the file
    with open(f"{output_dir}{file}", "r") as f:
        lines = f.readlines()
        processed_sentences, raw_sentences = preprocess_samples(lines)

    print(f"processed sentences: {processed_sentences}")
    print(f"raw sentences: {raw_sentences}")

    # then determine the topics of the sentences and the sentiment of the topics

    # then write the results to a file

# a sexy diagram would be nice too

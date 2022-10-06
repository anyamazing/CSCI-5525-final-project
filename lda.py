import pandas as pd
import json

filename = "AMAZON_FASHION_5.json"
with open(filename) as f:
    reviews_json = f.readlines()

reviews = []
for review_json in reviews_json:
    review = json.loads(review_json)
    if "overall" in review and "reviewText" in review:
        reviews.append(
            {
                "rating": review["overall"],
                "title": review["summary"],
                "text": review["reviewText"],
            }
        )

print(reviews)

df = pd.DataFrame.from_dict(reviews)
# jsonObj = pd.read_json(path_or_buf=file_path, lines=True)

print(df)

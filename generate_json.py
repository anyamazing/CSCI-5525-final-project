from data import generate_dataframe
import json

df = generate_dataframe(["Musical_Instruments_5.json"])
name = "music"

# count reviews per product
df.groupby("asin").count().sort_values("reviewText", ascending=False)

# remove products with less than 10 reviews
df = df.groupby("asin").filter(lambda x: len(x) >= 100)

product_groups = df.groupby("asin")


index = 0
test_products = []
train_products = []
for product_id, product_df in product_groups:
    if index < 100:
        # put each testing product into this format:
        """
        [
          {
            "entity_id": "...",
            "split": "dev",
            "reviews": [
              {
                "review_id": "...",
                "rating": 3,
                "sentences": [
                  "first sentence text",
                  "second sentence text",
                  ...
                ]
              },
              ...
            ],
            "summaries": {
              "general": [
                "reference summary 1 text",
                "reference summary 2 text",
                ...
              ],
              "aspect1": [...],
            }
          },
          ...
        ]
        """

        index += 1
        # create a list of reviews
        reviews = []

        # iterate over each review or the first 100 reviews
        for review_id, review in product_df.head(100).iterrows():
            # create a list of sentences
            sentences = []

            # iterate over each sentence
            for sentence in str(review["reviewText"]).split("."):
                # add the sentence to the list if it's not empty
                if sentence:
                    sentences.append(sentence.strip() + ".")

            # add the review to the list
            reviews.append(
                {
                    "review_id": str(review_id),
                    "rating": review["overall"],
                    "sentences": sentences,
                }
            )

        # add the product to the list
        test_products.append(
            {"entity_id": product_id, "reviews": reviews, "split": "dev"}
        )
    else:
        # put each training product into this format:
        """
        [
          {
            "entity_id": "...",
            "reviews": [
              {
                "review_id": "...",
                "rating": 3,
                "sentences": [
                  "first sentence text",
                  "second sentence text",
                  ...
                ]
              },
              ...
            ]
          },
          ...
        ]
        """

        # create a list of reviews
        reviews = []

        # iterate over each review or the first 100 reviews
        for review_id, review in product_df.head(100).iterrows():
            # create a list of sentences
            sentences = []

            # iterate over each sentence
            for sentence in str(review["reviewText"]).split("."):
                # add the sentence to the list if it's not empty
                if sentence:
                    sentences.append(sentence.strip() + ".")

            # add the review to the list
            reviews.append(
                {
                    "review_id": str(review_id),
                    "rating": review["overall"],
                    "sentences": sentences,
                }
            )

        # add the product to the list
        train_products.append(
            {
                "entity_id": product_id,
                # "split": "dev",
                "reviews": reviews,
                # "summaries": {"general": review["summary"].split(".")},
            }
        )

print(f"test products: {len(test_products)}, train products: {len(train_products)}")

# write the list to a file
with open(f"{name}_train.json", "w") as f:
    json.dump(train_products, f, indent=4)
with open(f"{name}_summ.json", "w") as f:
    json.dump(test_products, f, indent=4)

print(f"wrote to {name}_train.json and {name}_summ.json")

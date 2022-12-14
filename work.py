products = [
    "B0000AQRST",
    "B0002CZVK0",
    "B0002D0CQC",
    "B0002D08IE",
    "B0002E1NQY",
    "B0002GJ6FC",
    "B0002GYW4C",
    "B0002KZAKS",
    "B00063678K",
    "B00005ML71",
    "B0002CZVWI",
    "B0002D0CYE",
    "B0002DUPZU",
    "B0002E1NWI",
    "B0002GLCRC",
    "B0002GZBNS",
    "B0002KZE7C",
    "B0007GGUGA",
    "B000068NTU",
    "B0002CZW0Y",
    "B0002D0DWK",
    "B0002DUS8E",
    "B0002E1O3G",
    "B0002GLDQM",
    "B0002GZLZQ",
    "B0002M728Y",
    "B0007NQH98",
    "B000068NW8",
    "B0002CZW3G",
    "B0002D0E9M",
    "B0002DV6TO",
    "B0002E2GMY",
    "B0002GMGYA",
    "B0002GZM00",
    "B0002OOMU8",
    "B0007XE8YO",
    "B000068NW9",
    "B0002CZWXQ",
    "B0002D0HY4",
    "B0002DV7ZM",
    "B0002E2KPC",
    "B0002GMH7G",
    "B0002GZTT4",
    "B0002OOMW6",
    "B000068O4F",
    "B0002D0CA8",
    "B0002D0JX8",
    "B0002DV8AQ",
    "B0002E2OTE",
    "B0002GTZR6",
    "B0002H0A3S",
    "B0002OP7VQ",
    "B000068O35",
    "B0002D0CEO",
    "B0002D01K4",
    "B0002DVBJY",
    "B0002E2SA4",
    "B0002GW3Y8",
    "B0002H0JZ2",
    "B0002V8R5M",
    "B0001ARCFA",
    "B0002D0CGW",
    "B0002D017M",
    "B0002E1G5C",
    "B0002E3CHC",
    "B0002GWFEQ",
    "B0002H03YY",
    "B0006IQLF4",
    "B0002CZSJO",
    "B0002D0CH6",
    "B0002D02RQ",
    "B0002E1H9W",
    "B0002E3CK4",
    "B0002GX5NG",
    "B0002H04NE",
    "B0006IQLHM",
    "B0002CZUDS",
    "B0002D0CKI",
    "B0002D02SA",
    "B0002E1J3Q",
    "B0002E3D44",
    "B0002GXF8Q",
    "B0002H05BA",
    "B0006LOBA8",
    "B0002CZV46",
    "B0002D0CNA",
    "B0002D035C",
    "B0002E1NNC",
    "B0002F741Q",
    "B0002GXYVO",
    "B0002IHFVM",
    "B0006NDF8A",
    "B0002CZV82",
    "B0002D0COE",
    "B0002D07A8",
    "B0002E1NQE",
    "B0002FOBJY",
    "B0002GXZK4",
    "B0002II6V0",
    "B0006NDF76",
]

from data import generate_dataframe

df = generate_dataframe(["Musical_Instruments_5.json"])

# determine which asin in the list has the most reviews
# and print the asin and the number of reviews
df = df[df["asin"].isin(products)]
df = df.groupby("asin").count()
# get max overall
max_overall = df["overall"].max()
# get asin with max_overall.
max_asin = df[df["overall"] == max_overall].index[0]
print(max_asin, max_overall)
print(df)
print(df.head(1))

import pandas as pd
from datasets import Dataset

df = pd.read_csv("sentences.csv")
df = df.rename(columns={"sentences": "train"})
ds = Dataset.from_pandas(df)
print(ds)

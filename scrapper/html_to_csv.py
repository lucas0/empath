import pandas as pd

import glob, os, sys

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

os.chdir(cwd+"/pages")

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []
page_names = []
for file in glob.glob("*.html"):
    with open(file, "r+") as f:
        text = f.readlines()
        sents = [" ".join(tokenizer.tokenize(t.replace("\n"," "))).strip() for t in text]
        sents = [s for s in sents if len(s) > 10 and s[-1] != "?"]
        pages = [file]*len(sents)
        sentences += sents
        page_names += pages

df = pd.DataFrame(list(zip(sentences, page_names)), columns=["sentences", "pagenames"])
df.to_csv(cwd+"/sentences.csv", index=False)

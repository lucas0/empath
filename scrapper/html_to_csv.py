import pandas as pd

import glob, os, sys

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

os.chdir(cwd+"/pages")

import textacy, spacy
nlp = spacy.load("en_core_web_sm")
patterns = [{"POS": "AUX"}, {"POS": "VERB"}]

sentences = []
page_names = []
for idx,file in enumerate(glob.glob("*.html")):
    print(idx)
    with open(file, "r+") as f:
        #lines = f.readlines()
        text = f.read().replace('.\n', '.')
        text = text.replace('\n', '.')
        text = text.replace('..', '.')
        for s in nlp(text).sents:
            about_talk_doc = textacy.make_spacy_doc(s.text, lang="en_core_web_sm")
            verb_phrases = textacy.extract.token_matches(about_talk_doc, patterns=patterns)
            if len(list(verb_phrases)) > 0:
                sentences.append(s)
                page_names.append(file)

df = pd.DataFrame(list(zip(sentences, page_names)), columns=["sentences", "pagenames"])

print(len(df))
#remove duplicated sentences
df.drop_duplicates(inplace=True)
df.to_csv(cwd+"/verb_sentences.csv", index=False)
print(len(df))




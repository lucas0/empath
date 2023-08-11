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
max_length = 0
for idx,file in enumerate(glob.glob("*.html")):
    print(idx)
    with open(file, "r+") as f:
        #lines = f.readlines()
        text = f.read().replace('.\n', '.')
        text = text.replace('\n', '.')
        text = text.replace('..', '.')
        if len(text) > max_length:
            max_length = len(text) + 100
            nlp.max_length = max_length
        for s in nlp(text).sents:
            about_talk_doc = textacy.make_spacy_doc(s.text, lang="en_core_web_sm")
            verb_phrases = textacy.extract.token_matches(about_talk_doc, patterns=patterns)
            chunk = ""
            while len(chunk) < max_chunk_size:
            if len(list(verb_phrases)) > 0:
                sentences.append(s)
                page_names.append(file)

df = pd.DataFrame(list(zip(sentences, page_names)), columns=["sentences", "pagenames"])

df.drop_duplicates(inplace=True)
df.to_csv(cwd+"/verb_sentences.csv", index=False)
z = [str(e[0])+"|||"+e[1] for e in zip(df['sentences'].tolist(), df['pagenames'].tolist())]

#sentences = list(set(df['sentences'].tolist()))
with open(cwd+"/raw_verb_sentences.txt", "w+") as f:
    for s in z:
        f.write(str(s)+"<<<>>>")



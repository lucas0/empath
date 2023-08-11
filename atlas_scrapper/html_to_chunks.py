import pandas as pd

import glob, os, sys

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

os.chdir(cwd+"/pages")


chunks = []
page_names = []
max_length = 700
overlap_size = 50
l = len(glob.glob("*.html"))
for idx,file in enumerate(glob.glob("*.html")):
    print(idx+"/"+str(l))
    with open(file, "r+") as f:
        text = f.read()
        if len(text) > max_length:
            for a in range(0,len(text), max_length-overlap_size):
                chunks.append(text[a:a+max_length])
                page_names.append(file)
        else:
            chunks.append(text)
            page_names.append(file)

df = pd.DataFrame(list(zip(chunks, page_names)), columns=["chunks", "pagenames"])
df.drop_duplicates(inplace=True)
df.to_csv(cwd+"/chunks.csv", index=False)

z = [str(e[0])+"|||"+e[1] for e in zip(df['chunks'].tolist(), df['pagenames'].tolist())]

with open(cwd+"/chunks.txt", "w+") as f:
    for s in z:
        f.write(str(s)+"<<<>>>")



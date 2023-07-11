import warnings
warnings.filterwarnings('ignore')
import os,sys
import pandas as pd
import glob
from tqdm.auto import tqdm
import subprocess

this_script_name = sys.argv[0]
cwd = os.path.abspath(this_script_name+"/..")

# Step 1: loads the data and generate the embeddings from them
df = pd.read_csv(cwd+"/sentences.csv")

from sentence_transformers import SentenceTransformer
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")

# Step 2: connects to pinecone and uploads the data
import pinecone
pinecone.init(api_key="38a940d1-e29f-4459-9497-d9e6c91a265f", environment="us-west4-gcp-free")
index_name = "cisco-pages-qa"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=retriever.get_sentence_embedding_dimension(), metric='cosine')

index = pinecone.Index(index_name)
stats = index.describe_index_stats()
num_entries = stats['total_vector_count']

if num_entries != len(df):
    index.delete(deleteAll=True, namespace=index_name)
    batch_size = 64
    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i+batch_size, len(df))
        batch = df.iloc[i:i_end]
        emb = retriever.encode(batch["sentences"].tolist()).tolist()
        ids = [f"{idx}" for idx in range(i, i_end)]
        meta = batch.to_dict(orient="records")
        for idx, e in enumerate(meta):
            e_words = e["sentences"].split(" ")
            if len(e_words) > 5000:
                meta[idx]["sentences"] = ' '.join(e_words[:5000])
        to_upsert = list(zip(ids, emb, meta))
        _ = index.upsert(vectors=to_upsert)

def query_pinecone(query, top_k):
    xq = retriever.encode([query]).tolist()
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

context_size = 4096
def format_query(query, context):
    context = [f"{m['metadata']['sentences']}\n\n" for i,m in enumerate(context)]
    context = " ".join(context)
    q_l = len(query)
    max_cont = context_size-(q_l+20)
    context = context[:max_cont]
    query = f"Taking the following sentences as context: {context}. \n\nPlease answer: \"{query}\" \n\nANSWER: "
    return query

# Step 3: loads the model
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
model_name_or_path = "TheBloke/Chronos-Hermes-13B-SuperHOT-8K-fp16"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
config.max_position_embeddings = 4096
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
    config=config,
    trust_remote_code=True,
    device_map='auto')
pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
    )

ex1 = 'What are some of the Cisco Webex best features?'
ex2 = "What are the solutions for customer care provided by cisco?"
while True:
    print("===================")
    question = input("Input your question:\n>>> ")
    result = query_pinecone(question, top_k=10)
    query = format_query(question, result["matches"])
    output = pipe(query)[0]
    answer = output['generated_text']
    print(answer)
    with open(cwd+"/answers/"+question.replace(" ", "_"), "w+") as f:
        f.write(str(output))
    a = input("\nDo you want to see the sources?[y/n]: ")
    while a == "y":
        for e in result["matches"]:
            print(e["metadata"]+"\n\n")
            a = input("\nDo you want to see the next sources?[y/n]: ")

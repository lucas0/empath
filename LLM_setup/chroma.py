from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
import os, sys

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

#client = chromadb.PersistentClient(path=cwd+"/chroma_store")
client = chromadb.Client()
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
collection = client.get_or_create_collection(name="test", embedding_function=embedder.encode)

docs = ["Webex by Cisco is an American company that develops and sells web conferencing, videoconferencing and contact center as a service applications. It was founded as WebEx in 1995 and taken over by Cisco Systems in 2007. Its headquarters are in San Jose, California.", "With Webex Events, host interactive and engaging webinars with HD video and broadcast-quality audio for audiences up to 10,000.", "With Webex Training, lead live or on-demand training. Host breakouts, testing, and automated grading for over 1000 participants.", "Your Webex Meetings mobile app automatically connects to your video device when you enter the room so you can join your virtual meetings in a snap. The Webex Assistant provides real-time translations and transcripts – Accessible in your post meeting summary.", "Webex provides exceptional audio, video, and content sharing, including from digital white boards. Ensure fast and reliable video anywhere with the help of our global data centers. Custom layouts allow you to focus on what's important.", "With Webex Support, enjoy efficient customer service with remote desktop control and live chat."]

embeddings = embedder.encode(docs)
meta = [{"source":"https://www.cisco.com/c/en/us/products/conferencing/webex.html#~benefits"},
        {"source":"https://www.cisco.com/c/en/us/products/conferencing/webex.html#~benefits"},
        {"source":"https://www.cisco.com/c/en/us/products/conferencing/webex.html#~benefits"},
        {"source":"https://www.cisco.com/c/en/us/products/conferencing/webex.html#~benefits"},
        {"source":"https://www.cisco.com/c/en/us/products/conferencing/webex.html#~benefits"},
        {"source":"https://www.cisco.com/c/en/us/products/conferencing/webex.html#~benefits"},]
ids = ["id"+str(i) for i in range(len(docs))]

collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=meta,
        ids=ids)

model_name = "TheBloke/Chronos-Hermes-13B-SuperHOT-8k-fp16" 

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.15)


q1 = "What is Cisco Webex?"
retrieved_chroma = collection.query(query_texts=[q1], n_results=2)
prompt = "Please use the following documents as additional information in order to answer the question:\n\n"+"\n\n".join(retrieved_chroma["documents"][0])+"\n\n"+"### Question: "+q1
output = pipe(prompt)
print(output[0]["generated_text"])



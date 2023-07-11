# Empath

## Scrapper

Gets <body>  textual content from cisco customer centre pages.

## hermes

Runs the hermes model 13B, which is a fine-tuned version of llama 13B. 
This version uses contextual data by storing the sentences scrapped from cisco pages in a vector database in the form of embeddings and retrieving them via cosine similarity.

## finetune

finetune_hermes.py: finetunes the above-mentioned model with cisco sentences obtained with the scrapper.
run_finetuned_hermes.py: loads and runs the finetuned version of the hermes model.

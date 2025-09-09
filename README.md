# Automating the translation and mapping of Geospatial Application (GSA) data to HCAT

Using Mistral Nemo 12B and Retrieval Augmented Generation (RAG)

## Preprocessing

To prepare the database for 




## Running Inference
First we must precompute the context by RAG due to RAM constraints

To run generate_context: 
compulsory inputs
- path to GSA country file
- path to faiss-index generated earlier
- path to the metadata of the faiss-index generated earlier
- path to save the generated outputs

Sample terminal code (change the paths according to your system):
python generate_context.py 
  --country-csv /home/data/BW.csv 
  --faiss-index /home/automate_eurocrops/data/indices/agrovoc_agriprod_faiss.index 
  --metadata-pkl /home/automate_eurocrops/data/indices/agrovoc_agriprod_metadata.pkl 
  --output-pkl /home/automate_eurocrops/data/indices/BW_context.pkl


for inference
run 5 times with context
python inference.py   --mode ctx   --country-csv /home/eouser/tabVol/data/BW_hcat_mapped.csv   --hcat-xlsx /home/eouser/tabVol/data/HCAT4_new.xlsx   --contexts-pkl /home/eouser/tabVol/automate_eurocrops/data/indices/BW_context.pkl   --out-json /home/eouser/tabVol/automate_eurocrops/results/RAG_BW_results_ctx.json --runs 5 --consensus --consensus-out /home/eouser/tabVol/automate_eurocrops/results/RAG_BW_results_5runs_consensus.json

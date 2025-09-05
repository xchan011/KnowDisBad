

To run generate_context: compulsory inputs

python generate_context.py 
  --country-csv /home/eouser/tabVol/data/BW_hcat.csv 
  --faiss-index /home/eouser/tabVol/automate_eurocrops/data/indices/agrovoc_agriprod_faiss.index 
  --metadata-pkl /home/eouser/tabVol/automate_eurocrops/data/indices/agrovoc_agriprod_metadata.pkl 
  --output-pkl /home/eouser/tabVol/automate_eurocrops/data/indices/BW_context.pkl


for inference
run 5 times with context
python inference.py   --mode ctx   --country-csv /home/eouser/tabVol/data/BW_hcat_mapped.csv   --hcat-xlsx /home/eouser/tabVol/data/HCAT4_new.xlsx   --contexts-pkl /home/eouser/tabVol/automate_eurocrops/data/indices/BW_context.pkl   --out-json /home/eouser/tabVol/automate_eurocrops/results/RAG_BW_results_ctx.json --runs 5 --consensus --consensus-out /home/eouser/tabVol/automate_eurocrops/results/RAG_BW_results_5runs_consensus.json

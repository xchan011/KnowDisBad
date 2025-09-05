

To run generate_context: compulsory inputs

python generate_context.py \
  --country-csv /home/eouser/tabVol/data/BW_hcat.csv \
  --faiss-index /home/eouser/tabVol/automate_eurocrops/data/indices/agrovoc_agriprod_faiss.index \
  --metadata-pkl /home/eouser/tabVol/automate_eurocrops/data/indices/agrovoc_agriprod_metadata.pkl \
  --output-pkl /home/eouser/tabVol/automate_eurocrops/data/indices/BW_context.pkl

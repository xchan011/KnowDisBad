# Automating the translation and mapping of Geospatial Application (GSA) data to HCAT

Using Mistral Nemo 12B and Retrieval Augmented Generation (RAG)

## Preprocessing

To prepare the database for RAG.

Source of data:

[Agrovoc](https://www.fao.org/agrovoc/) root entries and its narrower concepts 
- crops
- plant products
- ornamental plants
- forest products
Definitions and synonyms, Relationships: includes, use of, produces, in English, German and French.

[Agriprod](https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/agriprod): Agriculture Products Dictionary of Eurostat
- In English, German and French
- Describes scope of, includes and excludes of each agriculture product

Both datasets are in excel, must be serialized-turned into prose like structure- for ingenstion to database.
See notebook serialize_data for details.

The next step requires generating the vector representations of the datasets and save them into a vector store. This is done in generate_faiss. Please note due to the large size of the generated index, it cannot be uploaded onto github.

To run generate_faiss:
- change the paths to the Agrovoc, Agriprod data in the file
- change the path to the files that will be created

If required, the dpl_trans notebook uses the deepl api to get the translations.

## Running Inference
First we must precompute the context by RAG due to RAM constraints: as running both the RAG and the prompt at the same time requires the loading of two LLMs at the same time consumes ~48GB of RAM.

To run generate_context: 
compulsory inputs
- path to GSA country file containing the list of crop names to be translated and mapped to HCAT
- path to faiss-index generated earlier
- path to the metadata of the faiss-index generated earlier
- path to save the generated outputs

Sample terminal code (change the paths according to your system):
python generate_context.py 
  --country-csv /home/automate_eurocrops/data/raw/BW.csv 
  --faiss-index /home/automate_eurocrops/data/indices/agrovoc_agriprod_faiss.index 
  --metadata-pkl /home/automate_eurocrops/data/indices/agrovoc_agriprod_metadata.pkl 
  --output-pkl /home/automate_eurocrops/data/indices/BW_context.pkl


For actual Inference:
compulsory inputs:
- path to GSA country file containing the list of crop names to be translated and mapped to HCAT
- path to HCAT excel file containing the list of HCAT names

optional inputs:
- path to saved context file generated in the previous step
- path to the saved deepl translated GSA crop names excel file

Modes available:
  - basic    : prompt uses only the original name
  - ctx      : prompt uses original name + context string
  - ctx-dpl  : prompt uses original name + context + per-row DeepL translation hint

Sample terminal code (change the paths according to your system):
modes: with context, runs 5 times, reprompts 5 times if validation failed
python inference.py --mode ctx --country-csv /home/automate_eurocrops/data/raw/BW.csv --hcat-xlsx /home/data/HCAT4_new.xlsx --contexts-pkl /home/automate_eurocrops/data/indices/BW_context.pkl --out-json-path /home/automate_eurocrops/results/RAG_dpl_BW_results.json --runs 5 --max-retries 5  --majority-result --majority-ou
t-path /home/automate_eurocrops/results/RAG_dpl_BW_results_consensus.json  

Both generate_context and inference comes with a notebook file to test out individual examples.

Final visualization of results can be seen in the notebook visualize_results.

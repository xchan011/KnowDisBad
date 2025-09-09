# 
"""
Script to create a vector store of the relevant datasets:
Code adapted from https://github.com/facebookresearch/faiss/tree/main/tutorial/python
This script makes use of FAISS to store the vector representations of AGROVOC and AGRIPROD data
- chunks the longer excel rows into shorter contexts -limit to 100 words per row
- turns the chunks batchwise into qwen3 8B embeddings and saved to FAISS index via cosine similarity
- save the index and the meta data (pkl). The original text chunks are required for context retrieval
"""

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize


# file paths and names
d_path= '/home/eouser/tabVol/automate_eurocrops/data/processed/'
data_name = 'agrovoc_agriprod_database.txt'
sv_path = '/home/eouser/tabVol/automate_eurocrops/data/indices/'
index_file_name = sv_path+'agrovoc_agriprod_faiss.index'
pkl_file_name= sv_path+'agrovoc_agriprod_metadata.pkl'


# Load
with open(d_path+data_name, 'r', encoding='utf-8') as f:
    rows_list = [line.strip() for line in f]

#Embeddings Model

em_model_id ='Qwen/Qwen3-Embedding-8B'
embedding_model = SentenceTransformer(em_model_id)


# need to convert the text into embeddings
def get_embeddings(texts,model, batch_size=8):
    """Generate embeddings using sentence-transformers"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(texts)} texts")
    
    return np.array(embeddings)

# split the text into smaller chunks if any exceeds 100 words
# keeping chunk size small as the longer the context the more memory consumed
def chunk_rows(row_strings, max_words=100):
    chunked_data = []
    
    for i, row in enumerate(row_strings):
        words = row.split()
        
        if len(words) <= max_words:
            # Keep short rows as-is
            chunked_data.append({
                'text': row,
                'source_row': i,
                'chunk_id': 0
            })
        else:
            # Split long rows by logical boundaries
            # Look for delimiters like commas, pipes, semicolons
            parts = row.split(', ')  # Adjust based on your delimiter
            
            current_chunk = []
            chunk_num = 0
            
            for part in parts:
                current_chunk.append(part)
                
                # Check if chunk is getting too long
                if len(' '.join(current_chunk).split()) > max_words:
                    if len(current_chunk) > 1:
                        # Save current chunk (minus last part)
                        chunk_text = ', '.join(current_chunk[:-1])
                        chunked_data.append({
                            'text': chunk_text,
                            'source_row': i,
                            'chunk_id': chunk_num
                        })
                        chunk_num += 1
                        current_chunk = [current_chunk[-1]]  # Start new chunk with last part
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ', '.join(current_chunk)
                chunked_data.append({
                    'text': chunk_text,
                    'source_row': i,
                    'chunk_id': chunk_num
                })
    
    return chunked_data

# Apply chunking
chunked_data = chunk_rows(rows_list)
print(f"Original rows: {len(rows_list)}")
print(f"After chunking: {len(chunked_data)}")

text_only = [row['text'] for row in chunked_data]

embeddings = get_embeddings(text_only,embedding_model, batch_size=8)
# normalize for cosine similarity
embeddings = normalize(embeddings, norm="l2", axis=1)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity

# Step 4: Add embeddings to index
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, index_file_name)

print(f"Saved FAISS index with {index.ntotal} vectors")

source_rows = [row['source_row'] for row in chunked_data]
chunk_ids = [row['chunk_id'] for row in chunked_data]

data_package = {
    'texts': text_only,
    'source_rows': source_rows,
    'chunk_ids': chunk_ids,
}

import pickle
with open(pkl_file_name, 'wb') as f:

    pickle.dump(data_package, f)
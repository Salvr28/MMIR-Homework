import sys
import os
import faiss
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, default_collate
from utils import load_config, get_device, generate_embeddings
from models import MultimodalModel
from dataset import ARCHDataset

# --- Config load ---
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = load_config(config_path)
print(f"Configs loaded from {config_path}")

# --- Device load ---
device = get_device()

# --- Output Path ---
output_dir = config['embeddings']['output_dir']
os.makedirs(output_dir, exist_ok=True) 

image_embeddings_path = os.path.join(output_dir, config['embeddings']['image_embeddings_file'])
text_embeddings_path = os.path.join(output_dir, config['embeddings']['text_embeddings_file'])
metadata_path = os.path.join(output_dir, config['embeddings']['metadata_file'])
faiss_index_path = os.path.join(output_dir, config['embeddings']['faiss_index_file'])

# --- Custom collate function ---
def custom_collate_fn(batch):
    # Filter all none samples
    batch = [item for item in batch if item is not None]
    if not batch: 
        print("Warning: All samples in a batch were None. Skipping this batch.") # Debugging
        return None 
    return default_collate(batch) 


# --- Main function for indexing ---
def main_generate_and_index():
    print("\n--- Starting Embedding Generation and FAISS Indexing ---")

    image_encoder_name = config['model']['image_encoder']['name'].replace("/", "_").replace("-", "_")
    text_encoder_name = config['model']['text_encoder']['name'].replace("/", "_").replace("-", "_")

    # --- Embeddings output paths ---
    output_dir = config['embeddings']['output_dir']
    os.makedirs(output_dir, exist_ok=True) 

    # File names based on models names
    image_embeddings_filename = f"image_embeddings_{image_encoder_name}_{text_encoder_name}.npy"
    text_embeddings_filename = f"text_embeddings_{image_encoder_name}_{text_encoder_name}.npy"
    metadata_filename = f"metadata_{image_encoder_name}_{text_encoder_name}.json"
    faiss_image_index_filename = f"faiss_index_images_{image_encoder_name}_{text_encoder_name}.bin"
    faiss_text_index_filename = f"faiss_index_texts_{image_encoder_name}_{text_encoder_name}.bin"

    image_embeddings_path = os.path.join(output_dir, image_embeddings_filename)
    text_embeddings_path = os.path.join(output_dir, text_embeddings_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)
    faiss_image_index_path = os.path.join(output_dir, faiss_image_index_filename)
    faiss_text_index_path = os.path.join(output_dir, faiss_text_index_filename)

    # --- Model loading ---
    image_embeddings_np = None
    text_embeddings_np = None
    all_metadata = None

    # If embeddings already exist, loads them
    if os.path.exists(image_embeddings_path) and os.path.exists(text_embeddings_path) and os.path.exists(metadata_path):
        print("Existing embeddings and metadata found. Loading them...")
        image_embeddings_np = np.load(image_embeddings_path)
        text_embeddings_np = np.load(text_embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
        print("Embeddings and metadata loaded successfully.")
    else:
        print("Loading dataset...")
        full_dataset = ARCHDataset(config, target_sets=["pubmed_set", "books_set"]) 
        
        BATCH_SIZE = 32 
        num_workers = 3 
        
        full_dataloader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False, 
            collate_fn=custom_collate_fn
        )

        print("Initializing MultimodalModel...")
        model = MultimodalModel(config)
        model.to(device)

        # --- Embedding generation ---
        print("Generating embeddings for the entire dataset...")
        image_embeddings_np, text_embeddings_np, all_metadata = generate_embeddings(model, full_dataloader, device)

        # --- Embedding and metadata saving ---
        print("Saving generated embeddings and metadata...")
        np.save(image_embeddings_path, image_embeddings_np)
        np.save(text_embeddings_path, text_embeddings_np)
        
        # Metadata are saved like json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4)
        
        print(f"Image embeddings saved to: {image_embeddings_path}")
        print(f"Text embeddings saved to: {text_embeddings_path}")
        print(f"Metadata saved to: {metadata_path}")

    # --- Building Faiss index and saving ---
    print("Building FAISS index...")
     

    image_embeddings_np = image_embeddings_np.astype('float32')
    text_embeddings_np = text_embeddings_np.astype('float32')
    
    embedding_dim = image_embeddings_np.shape[1]

    #---- Index Implementation ----
    # M: Number of bidirectional connection for each node
    M = config['faiss']['hnsw']['M'] if 'hnsw' in config['faiss'] and 'M' in config['faiss']['hnsw'] else 32

    # efConstruction: List dimension of candidates during graph building
    efConstruction = config['faiss']['hnsw']['efConstruction'] if 'hnsw' in config['faiss'] and 'efConstruction' in config['faiss']['hnsw'] else 100

    # efSearch: List dimension of candidates during search
    efSearch = config['faiss']['hnsw']['efSearch'] if 'hnsw' in config['faiss'] and 'efSearch' in config['faiss']['hnsw'] else 50

    print(f"Using HNSW parameters: M={M}, efConstruction={efConstruction}, efSearch={efSearch}")

    # Image index
    image_index_hnsw = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
    image_index_hnsw.efConstruction = efConstruction 
    print("Adding image embeddings to HNSW index...")
    image_index_hnsw.add(image_embeddings_np) 
    image_index_hnsw.efSearch = efSearch 
    faiss.write_index(image_index_hnsw, faiss_image_index_path)
    print(f"Image IndexHNSWFlat saved to: {faiss_image_index_path}")


    # Text index
    text_index_hnsw = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
    text_index_hnsw.efConstruction = efConstruction 
    print("Adding text embeddings to HNSW index...")
    text_index_hnsw.add(text_embeddings_np)
    text_index_hnsw.efSearch = efSearch 
    faiss.write_index(text_index_hnsw, faiss_text_index_path)
    print(f"Text IndexHNSWFlat saved to: {faiss_text_index_path}")

    print("\n--- Embedding Generation and FAISS Indexing Complete ---")

# --- Pipeline execution ---
if __name__ == '__main__':
    main_generate_and_index()
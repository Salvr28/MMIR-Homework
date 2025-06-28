import yaml
import faiss
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
from models import MultimodalModel
from dataset import ARCHDataset
import numpy as np

def load_config(config_path="config.yaml"):
    """
    Returns configurations from config file as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_device():
    """
    Determines the appropriate device (CPU or GPU) for PyTorch.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available(): # Mac
        device = torch.device("mps")
        print("Using Apple Silicon MPS (GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def generate_embeddings(model, dataloader, device):
    """
    Generates image and text embeddings for a given model and dataloader.

    Args:
        model (torch.nn.Module): The multimodal model to use for embedding generation.
        dataloader (torch.utils.data.DataLoader): DataLoader providing image and text data.
        device (torch.device): The device (CPU/GPU) to perform computations on.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - image_embeddings_np (numpy.ndarray): All generated image embeddings.
               - text_embeddings_np (numpy.ndarray): All generated text embeddings.
        list: A list of dicts containing metadata for each sample (uuid, source_set, etc.).
              This metadata list will be in the same order as the embeddings.
    """

    # Inference mode
    model.eval()
    model.to(device)

    all_image_embeddings = []
    all_text_embeddings = []
    all_metadata = []

    print(f"Generating embeddings on {device}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating Embeddings'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            img_embeddings, txt_embeddings = model(images, input_ids, attention_mask)

            all_image_embeddings.append(img_embeddings.cpu().numpy())
            all_text_embeddings.append(txt_embeddings.cpu().numpy())

            # Gathering metadata
            for i in range(len(batch['uuid'])):
                all_metadata.append({
                    'uuid': batch['uuid'][i],
                    'original_text': batch['original_text'][i], 
                    'caption': batch['caption'][i],
                    'image_path': batch['image_path'][i],
                    'source_set': batch['source_set'][i],
                    'letter': batch['letter'][i]               
                })

        # Concat all embeddings in one array np
        image_embeddings_np = np.vstack(all_image_embeddings)
        text_embeddings_np = np.vstack(all_text_embeddings)

    print(f"Generated {len(image_embeddings_np)} image embeddings and {len(text_embeddings_np)} text embeddings.")

    return image_embeddings_np, text_embeddings_np, all_metadata

def calculate_cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two 1D torch tensors or numpy arrays.
    Assumes vectors are already L2 normalized if using dot product for cosine.
    """

    if isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):   
        vec1_t = torch.from_numpy(vec1)
        vec2_t = torch.from_numpy(vec2)
    else:
        vec1_t = vec1.float()
        vec2_t = vec2.float()

    # If vectors are already normalized similarity is dot product
    # Otherwise: F.cosine_similarity(vec1_t, vec2_t, dim=0)
    return  torch.dot(vec1_t, vec2_t).item()

def show_image(image_path, title=""):
    """
    Displays an image from a given path using matplotlib.
    """
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"Image not found at: {image_path}")

def perform_faiss_search(query_embedding, faiss_index, k=10):
    """
    Performs a similarity search using a FAISS index.

    Args:
        query_embedding (numpy.ndarray): The embedding of the query,
                                         must be a 2D array (e.g., [1, embedding_dim]).
        faiss_index (faiss.Index): The loaded FAISS index.
        k (int): The number of nearest neighbors to retrieve.

    Returns:
        tuple: A tuple containing:
               - distances (numpy.ndarray): Distances/similarities of the top-k results.
               - indices (numpy.ndarray): Original indices of the top-k results in the FAISS index.
    """
    # Check if query embeddings is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # FAISS wants float32
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    distances, indices = faiss_index.search(query_embedding, k)
    
    # Returns 1D arrays for simplicity
    return distances.squeeze(), indices.squeeze()

# ------------------------ Retrieval Evaluation -------------------------------------

def load_data_for_evaluation(config_path):
    """
    Loads image and text embeddings, metadata, and FAISS indices
    generated by the embeddings.py module.
    """
    config = load_config(config_path)
    output_dir = config['embeddings']['output_dir']

    # Files names based on models names
    image_encoder_name = config['model']['image_encoder']['name'].replace("/", "_").replace("-", "_")
    text_encoder_name = config['model']['text_encoder']['name'].replace("/", "_").replace("-", "_")

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

    print(f"Loading image embeddings from: {image_embeddings_path}")
    image_embeddings = np.load(image_embeddings_path)
    print(f"Loading text embeddings from: {text_embeddings_path}")
    text_embeddings = np.load(text_embeddings_path)
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loading FAISS image index from: {faiss_image_index_path}")
    image_index = faiss.read_index(faiss_image_index_path)
    print(f"Loading FAISS text index from: {faiss_text_index_path}")
    text_index = faiss.read_index(faiss_text_index_path)

    image_embeddings = image_embeddings.astype('float32')
    text_embeddings = text_embeddings.astype('float32')

    return image_embeddings, text_embeddings, metadata, image_index, text_index, config

def load_data_for_rag_evaluation(config_path):
    """
    Loads pre-generated embeddings (ARCHDataset images and texts),
    FAISS indices, the multimodal model (for transformations and embeddings)
    and ARCHDataset metadata, specifically for RAG evaluation.
    """
    # Use the existing function to load common data
    image_embeddings_arch, text_embeddings_arch, arch_metadata, \
        _, text_index_arch, config = load_data_for_evaluation(config_path)
    
    # --- Additional parts specific to RAG evaluation ---
    device = get_device()
    model = MultimodalModel(config).to(device)
    model.eval() 

    # Initializes a "dummy" ARCHDataset instance to get the correct image_processor
    print("Initializing dummy ARCHDataset to get image pre-processor...")
    # Using a dummy target_sets as we don't load data here
    dummy_arch_dataset = ARCHDataset(config, target_sets="pubmed_set") 
    
    # Get the specific processor from the dataset
    image_processor_for_query = None
    if dummy_arch_dataset.timm_image_processor is not None:
        image_processor_for_query = dummy_arch_dataset.timm_image_processor
    elif dummy_arch_dataset.image_processor is not None:
        image_processor_for_query = dummy_arch_dataset.image_processor
    else:
        raise ValueError("ARCHDataset failed to initialize image processor. Cannot proceed.")

    return image_embeddings_arch, text_embeddings_arch, arch_metadata, \
           text_index_arch, model, image_processor_for_query, config, device


def calculate_f1(precision, recall):
    """Calcolates F1-score."""
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_average_precision(retrieved_ranks, num_relevant_total):
    """
    Calculates Average Precision for a single query.
    Args:
        retrieved_ranks (list): List of tuples (rank, is_relevant) for retrieved documents.
                                e.g., [(1, 1), (2, 0), (3, 1)] where 1=relevant, 0=not_relevant
        num_relevant_total (int): Total number of relevant items in the dataset for the current query.
    Returns:
        float: Average Precision.
    """
    if num_relevant_total == 0:
        return 0.0

    sum_precisions = 0.0
    relevant_count = 0
    for i, (rank, is_relevant) in enumerate(retrieved_ranks):
        if is_relevant:
            relevant_count += 1
            precision_at_k = relevant_count / (i + 1) # i+1 is the current rank
            sum_precisions += precision_at_k
    
    return sum_precisions / num_relevant_total

def calculate_ndcg(retrieved_ranks, num_relevant_total):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) for a single query.
    We assume binary relevance (1 for relevant, 0 for not relevant).
    Args:
        retrieved_ranks (list): List of tuples (rank, is_relevant) for retrieved documents.
                                e.g., [(1, 1), (2, 0), (3, 1)] where 1=relevant, 0=not_relevant
        num_relevant_total (int): Total number of relevant items in the dataset for the current query.
    Returns:
        float: NDCG score.
    """
    if not retrieved_ranks:
        return 0.0

    dcg = 0.0
    for i, (rank, is_relevant) in enumerate(retrieved_ranks):
        if is_relevant:
            dcg += 1.0 / np.log2(i + 1 + 1) # i+1 is rank, so i+1+1 for log2(rank+1)

    # Calculate IDCG (Ideal DCG)
    # For binary relevance, IDCG is sum of 1/log2(i+1) for first num_relevant_total items
    idcg = 0.0
    for i in range(min(len(retrieved_ranks), num_relevant_total)): # Only sum up to k or total relevant, whichever is smaller
        idcg += 1.0 / np.log2(i + 1 + 1)
    
    if idcg == 0: # No relevant items or only non-relevant retrieved
        return 0.0
    
    return dcg / idcg

def calculate_mean_reciprocal_rank(retrieved_ranks):
    """
    Calculates the Reciprocal Rank for a single query.
    Args:
        retrieved_ranks (list): List of tuples (rank, is_relevant) for retrieved documents.
                                e.g., [(1, 1), (2, 0), (3, 1)] where 1=relevant, 0=not_relevant
    Returns:
        float: Reciprocal Rank.
    """
    for i, (rank, is_relevant) in enumerate(retrieved_ranks):
        if is_relevant:
            return 1.0 / (i + 1) # i+1 is the current rank
    return 0.0 # No relevant item found





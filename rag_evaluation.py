import sys
import time
import os
import faiss
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from rouge import Rouge 
import torch
import torchvision.transforms as transforms 
from google.api_core import exceptions
import google.generativeai as genai
from utils import load_config, get_device, load_data_for_rag_evaluation
from models import MultimodalModel 
from dataset import ARCHDataset 

config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = load_config(config_file_path)

# --- LLM Api configuration ---
GEMINI_API_KEY = str(config['rag']['model_api'])
genai.configure(api_key=GEMINI_API_KEY)


# --- Variables for quota control ---
requests_count = 0
requests_per_batch = 5  # Number of requests before a pause
pause_duration = 90     # Pause duration in seconds (1 and a half minutes)

def retrieve_context_for_image(query_image_embedding: np.ndarray, text_index: faiss.Index, arch_metadata: list, k: int = 10) -> list[str]:
    """
    Retrieves the top k texts most similar to a query image using the FAISS index.
    Returns the original texts from the ARCHDataset.
    """
    D, I = text_index.search(query_image_embedding.reshape(1, -1), k)
    retrieved_indices = I[0] 

    retrieved_texts = []
    for idx in retrieved_indices:
        if 0 <= idx < len(arch_metadata):
            retrieved_texts.append(arch_metadata[idx]['caption']) 
    return retrieved_texts

def prepare_gemini_prompt(image: Image.Image, retrieved_texts: list[str]) -> list:
    """
    Prepares the prompt for the Gemini API, including the image and retrieved texts.
    """
    context_str = "\n".join([f"- {text}" for text in retrieved_texts])
    
    prompt_parts = [
        image,
        f"Based on the following retrieved medical and histological texts, describe this pathological image. Focus on relevant details, scientific observations, and factual information presented in the texts and evident in the image, especially concerning cellular structures, tissue morphology, and potential pathological features.\n\nRetrieved Medical Context:\n{context_str}\n\nPathological Description:",
    ]
    return prompt_parts

def generate_description_with_gemini(model_gemini, image: Image.Image, retrieved_texts: list[str]) -> str:
    """
    Sends the query to the Gemini API and returns the generated description.
    """
    try:
        prompt = prepare_gemini_prompt(image, retrieved_texts)
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        return ""

def load_reference_queries(path: str) -> dict:
    """
    Loads reference queries and gold standard descriptions from a JSON file.
    """
    if not os.path.exists(path):
        print(f"Error: Reference queries file not found at {path}. Please create it.")
        print("Expected format: {'query_id': {'image_path': 'path/to/image.jpg', 'ground_truth_description': 'Your detailed description.'}}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Main function for rag system evaluation
def main_rag_evaluation():

    global requests_count

    config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    image_embeddings_arch, text_embeddings_arch, arch_metadata, \
    text_index_arch, arch_multimodal_model, image_processor_for_query, \
    config, device = load_data_for_rag_evaluation(config_file_path)

    gemini_model = genai.GenerativeModel('gemini-2.0-flash') 
    
    reference_queries = load_reference_queries(config['rag']['queries_path'])
    if not reference_queries:
        return

    rouge = Rouge() 

    all_generated_descriptions = []
    all_reference_descriptions = []

    print(f"\n--- Starting RAG Evaluation for {len(reference_queries)} queries ---")
    k_retrieval = 10 

    for i, query_data in tqdm(enumerate(reference_queries), total=len(reference_queries), desc="Evaluating RAG Queries"):

        query_id = f"query_{i}" 
        if query_data['source'] == 'books_set':
            image_path = os.path.join(config['dataset']['book_dir'], 'images',query_data['image_path'])
        else:
            image_path = os.path.join(config['dataset']['pubmed_dir'], 'images',query_data['image_path'])
        ground_truth_description = query_data['ground_truth_description']

        if not os.path.exists(image_path):
            print(f"Warning: Image not found for query {query_id} at {image_path}. Skipping.")
            continue
        
        try:
            # Load the image
            image_pil = Image.open(image_path).convert("RGB")
            
            # Load corrispective precalculated embedding
            query_idx_in_arch_dataset = -1
            for idx, meta_item in enumerate(arch_metadata):
                if os.path.abspath(meta_item['image_path']) == os.path.abspath(image_path):
                    query_idx_in_arch_dataset = idx
                    break
            
            if query_idx_in_arch_dataset == -1:
                # Fallback: se per qualche ragione l'immagine di query non è nei metadati,
                # calcola l'embedding al volo usando il processore ottenuto da ARCHDataset
                print(f"Warning: Query image {image_path} not found in ARCHDataset metadata. Calculating embedding on the fly.")
                
                # Applica le trasformazioni corrette ottenute da ARCHDataset
                if isinstance(image_processor_for_query, transforms.Compose): # Timm returns Compose
                    processed_image_tensor = image_processor_for_query(image_pil).unsqueeze(0).to(device)
                elif hasattr(image_processor_for_query, '__call__'): # HF AutoFeatureExtractor/Processor
                    # Assumi che riceva PIL Image e restituisca un tensore batchato
                    processed_image_tensor = image_processor_for_query(images=image_pil, return_tensors="pt").pixel_values.to(device)
                else:
                    raise ValueError("Unsupported image processor type for on-the-fly embedding calculation.")

                with torch.no_grad():
                    query_image_embedding = arch_multimodal_model.image_encoder(processed_image_tensor).cpu().numpy()
                    # Normalizza l'embedding, come fatto nel forward del MultimodalModel
                    query_image_embedding = query_image_embedding / np.linalg.norm(query_image_embedding)

            else:
                # Usa l'embedding pre-calcolato per la query image se trovata
                query_image_embedding = image_embeddings_arch[query_idx_in_arch_dataset]
                # Anche se sono pre-calcolati, rinormalizzo per sicurezza, dato che il forward del MM lo fa
                query_image_embedding = query_image_embedding / np.linalg.norm(query_image_embedding)


            retrieved_texts = retrieve_context_for_image(
                query_image_embedding, 
                text_index_arch, 
                arch_metadata, 
                k=k_retrieval
            )

            # --- RATE LIMITING LOGIC ---
            # Gemini has a limit rate of 5 queries per minute
            if requests_count >= requests_per_batch:
                print(f"\n--- Quota limit reached ({requests_per_batch} requests). Pausing for {pause_duration} seconds... ---")
                time.sleep(pause_duration)
                requests_count = 0 
            
            try:
                # Query and increments count of queries per minute
                generated_description = generate_description_with_gemini(gemini_model, image_pil, retrieved_texts)
                requests_count += 1 
            
            # Error for quota limitation
            except exceptions.ResourceExhausted as e: 
                print(f"Caught 429 (Quota Exceeded) for query {query_id}. This request will not count towards requests_per_batch until successful.")
                generated_description = "Failed to generate due to API quota limits (429 encountered)."
            except Exception as e: 
                print(f"An unexpected error occurred with Gemini API for query {query_id}: {e}")
                generated_description = f"Error during generation: {e}"
                requests_count += 1

            # Check if description is valid for rouge
            if not generated_description.strip(): 
                generated_description = "Description generation failed or was empty."

            all_generated_descriptions.append(generated_description)
            all_reference_descriptions.append(ground_truth_description)
           
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            import traceback
            traceback.print_exc() 
            continue

    if not all_generated_descriptions:
        print("No descriptions were generated. Cannot perform ROUGE evaluation.")
        return

    print("\nCalculating ROUGE scores...")
    scores = rouge.get_scores(all_generated_descriptions, all_reference_descriptions, avg=True)

    print("\n--- ROUGE Scores (Average across all queries) ---")
    for metric, values in scores.items():
        print(f"{metric.upper()}:")
        print(f"  Precision: {values['p']:.4f}")
        print(f"  Recall:    {values['r']:.4f}")
        print(f"  F1-Score:  {values['f']:.4f}")

    print("\n--- RAG Evaluation Complete ---")

# ---- Rag pipeline execution ----
if __name__ == '__main__':
    main_rag_evaluation()
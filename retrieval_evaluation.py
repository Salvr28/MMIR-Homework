import sys
import os
import faiss
import numpy as np
import json
from tqdm import tqdm
from utils import load_config, load_data_for_evaluation, calculate_cosine_similarity, calculate_average_precision, calculate_f1, calculate_mean_reciprocal_rank, calculate_ndcg 

def evaluate_system(image_embeddings, text_embeddings, metadata, image_index, text_index, k=10, similarity_threshold=0.8):
    """
    Performs system evaluation in Image-to-Text and Text-to-Image modes.

    Args:
        image_embeddings (np.ndarray): Array of image embeddings.
        text_embeddings (np.ndarray): Array of text embeddings.
        metadata (list): List of metadata associated with each embedding.
        image_index (faiss.Index): FAISS index for image embeddings.
        text_index (faiss.Index): FAISS index for text embeddings.
        k (int): Number of items to retrieve per query (and for @k metric calculation).
        similarity_threshold (float): Cosine similarity threshold to consider an item relevant.

    Returns:
        dict: Dictionary containing average metrics for each mode.
    """
    num_samples = len(metadata) # Number of image-text couples
    
    # --- Image to Text evaluation ---
    print(f"\n--- Evaluating Image to Text Retrieval (k={k}, threshold={similarity_threshold}) ---")
    total_precision_it = 0.0
    total_recall_it = 0.0
    total_f1_it = 0.0
    total_ap_it = 0.0 # Average Precision
    total_ndcg_it = 0.0 # NDCG
    total_mrr_it = 0.0 # Mean Reciprocal Rank
    num_valid_queries_it = 0

    for i in tqdm(range(num_samples), desc="Image to Text"):
        query_image_embedding = image_embeddings[i]
        
        # Determine the set of ACTUALLY RELEVANT items for this query in the entire dataset.
        true_relevant_item_indices_for_query = set()
        for j in range(num_samples):
            # A relevant item is text T_j if image I_j is sufficiently similar to I_q
            if calculate_cosine_similarity(query_image_embedding, image_embeddings[j]) >= similarity_threshold:
                true_relevant_item_indices_for_query.add(j)
        
        # If there are no relevant items in the dataset for this query according to the threshold, we skip it for metrics.
        if not true_relevant_item_indices_for_query:
            # print(f"Warning: No relevant items found for query {i} in Image to Text mode with threshold {similarity_threshold}. Skipping.")
            continue

        # Top k results from faiss index
        D, I = text_index.search(query_image_embedding.reshape(1, -1), k)
        retrieved_text_indices_ranked = set(I[0]) # set for efficient lookup

        # Calculates true positive and false negative
        relevant_retrieved_count_it = 0 # TP
        retrieved_items_for_ap_ndcg = []
        
        for rank_idx, retrieved_idx in enumerate(retrieved_text_indices_ranked):
            is_relevant = 1 if retrieved_idx in true_relevant_item_indices_for_query else 0
            retrieved_items_for_ap_ndcg.append((rank_idx + 1, is_relevant)) # (rank, is_relevant)
            if is_relevant:
                relevant_retrieved_count_it += 1
        
        # Precision@k: (TP / k)
        precision_q_it = relevant_retrieved_count_it / k if k > 0 else 0.0
        
        # Recall@k: (TP / (TP + FN)) = (TP / Tot relevant)
        recall_q_it = relevant_retrieved_count_it / len(true_relevant_item_indices_for_query)
        
        # F1-Score@k
        f1_q_it = calculate_f1(precision_q_it, recall_q_it)

        # Average Precision (AP)
        ap_q_it = calculate_average_precision(retrieved_items_for_ap_ndcg, len(true_relevant_item_indices_for_query))

        # NDCG@k
        ndcg_q_it = calculate_ndcg(retrieved_items_for_ap_ndcg, len(true_relevant_item_indices_for_query))

        # Mean Reciprocal Rank (MRR)
        mrr_q_it = calculate_mean_reciprocal_rank(retrieved_items_for_ap_ndcg)


        # Updates totals
        total_precision_it += precision_q_it
        total_recall_it += recall_q_it
        total_f1_it += f1_q_it
        total_ap_it += ap_q_it
        total_ndcg_it += ndcg_q_it
        total_mrr_it += mrr_q_it
        num_valid_queries_it += 1

    # Final averages
    avg_precision_it = total_precision_it / num_valid_queries_it if num_valid_queries_it > 0 else 0.0
    avg_recall_it = total_recall_it / num_valid_queries_it if num_valid_queries_it > 0 else 0.0
    avg_f1_it = total_f1_it / num_valid_queries_it if num_valid_queries_it > 0 else 0.0
    avg_ap_it = total_ap_it / num_valid_queries_it if num_valid_queries_it > 0 else 0.0
    avg_ndcg_it = total_ndcg_it / num_valid_queries_it if num_valid_queries_it > 0 else 0.0
    avg_mrr_it = total_mrr_it / num_valid_queries_it if num_valid_queries_it > 0 else 0.0

    print(f"   Average Precision@{k} (Image to Text): {avg_precision_it:.4f}")
    print(f"   Average Recall@{k} (Image to Text): {avg_recall_it:.4f}")
    print(f"   Average F1-Score@{k} (Image to Text): {avg_f1_it:.4f}")
    print(f"   Mean Average Precision (MAP)@{k} (Image to Text): {avg_ap_it:.4f}")
    print(f"   Normalized Discounted Cumulative Gain (NDCG)@{k} (Image to Text): {avg_ndcg_it:.4f}")
    print(f"   Mean Reciprocal Rank (MRR) (Image to Text): {avg_mrr_it:.4f}")

    # --- Valutazione Text to Image ---
    print(f"\n--- Evaluating Text to Image Retrieval (k={k}, threshold={similarity_threshold}) ---")
    total_precision_ti = 0.0
    total_recall_ti = 0.0
    total_f1_ti = 0.0
    total_ap_ti = 0.0
    total_ndcg_ti = 0.0
    total_mrr_ti = 0.0
    num_valid_queries_ti = 0

    for i in tqdm(range(num_samples), desc="Text to Image"):
        query_text_embedding = text_embeddings[i]

        # Determine the set of ACTUALLY RELEVANT items for this query in the entire dataset.
        true_relevant_item_indices_for_query = set()
        # Assuming: (Text_q, Image_j) is relevant if Sim(Text_q, Text_j) >= threshold
        for j in range(num_samples):
            if calculate_cosine_similarity(query_text_embedding, text_embeddings[j]) >= similarity_threshold:
                true_relevant_item_indices_for_query.add(j)
        
        if not true_relevant_item_indices_for_query:
            continue

        # faiss index retrieval
        D, I = image_index.search(query_text_embedding.reshape(1, -1), k)
        retrieved_image_indices_ranked = I[0]

        # Prepare for metric calculation
        relevant_retrieved_count_ti = 0
        retrieved_items_for_ap_ndcg = []
        
        for rank_idx, retrieved_idx in enumerate(retrieved_image_indices_ranked):
            is_relevant = 1 if retrieved_idx in true_relevant_item_indices_for_query else 0
            retrieved_items_for_ap_ndcg.append((rank_idx + 1, is_relevant))
            if is_relevant:
                relevant_retrieved_count_ti += 1

        # Precision@k
        precision_q_ti = relevant_retrieved_count_ti / k if k > 0 else 0.0
        
        # Recall@k
        recall_q_ti = relevant_retrieved_count_ti / len(true_relevant_item_indices_for_query)
        
        # F1-Score@k
        f1_q_ti = calculate_f1(precision_q_ti, recall_q_ti)

        # Average Precision (AP)
        ap_q_ti = calculate_average_precision(retrieved_items_for_ap_ndcg, len(true_relevant_item_indices_for_query))

        # NDCG@k
        ndcg_q_ti = calculate_ndcg(retrieved_items_for_ap_ndcg, len(true_relevant_item_indices_for_query))

        # Mean Reciprocal Rank (MRR)
        mrr_q_ti = calculate_mean_reciprocal_rank(retrieved_items_for_ap_ndcg)

        # Updates totals
        total_precision_ti += precision_q_ti
        total_recall_ti += recall_q_ti
        total_f1_ti += f1_q_ti
        total_ap_ti += ap_q_ti
        total_ndcg_ti += ndcg_q_ti
        total_mrr_ti += mrr_q_ti
        num_valid_queries_ti += 1

    # Final avereges
    avg_precision_ti = total_precision_ti / num_valid_queries_ti if num_valid_queries_ti > 0 else 0.0
    avg_recall_ti = total_recall_ti / num_valid_queries_ti if num_valid_queries_ti > 0 else 0.0
    avg_f1_ti = total_f1_ti / num_valid_queries_ti if num_valid_queries_ti > 0 else 0.0
    avg_ap_ti = total_ap_ti / num_valid_queries_ti if num_valid_queries_ti > 0 else 0.0
    avg_ndcg_ti = total_ndcg_ti / num_valid_queries_ti if num_valid_queries_ti > 0 else 0.0
    avg_mrr_ti = total_mrr_ti / num_valid_queries_ti if num_valid_queries_ti > 0 else 0.0

    print(f"   Average Precision@{k} (Text to Image): {avg_precision_ti:.4f}")
    print(f"   Average Recall@{k} (Text to Image): {avg_recall_ti:.4f}")
    print(f"   Average F1-Score@{k} (Text to Image): {avg_f1_ti:.4f}")
    print(f"   Mean Average Precision (MAP)@{k} (Text to Image): {avg_ap_ti:.4f}")
    print(f"   Normalized Discounted Cumulative Gain (NDCG)@{k} (Text to Image): {avg_ndcg_ti:.4f}")
    print(f"   Mean Reciprocal Rank (MRR) (Text to Image): {avg_mrr_ti:.4f}")
    
    print("\n--- Evaluation Complete ---")
    
    return {
        "image_to_text": {
            "precision": avg_precision_it, 
            "recall": avg_recall_it, 
            "f1": avg_f1_it,
            "map": avg_ap_it,
            "ndcg": avg_ndcg_it,
            "mrr": avg_mrr_it
        },
        "text_to_image": {
            "precision": avg_precision_ti, 
            "recall": avg_recall_ti, 
            "f1": avg_f1_ti,
            "map": avg_ap_ti,
            "ndcg": avg_ndcg_ti,
            "mrr": avg_mrr_ti
        }
    }

if __name__ == '__main__':
    # config path
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    print("Loading data for evaluation...")
    # loads necessary data
    image_embs, text_embs, meta, img_idx, text_idx, loaded_config = load_data_for_evaluation(config_file_path)

    # Evaluation parameters
    k_value = 10 
    sim_threshold = 0.6 # similarity treshold

    print(f"Loaded {len(image_embs)} image embeddings and {len(text_embs)} text embeddings.")
    print(f"Starting evaluation with k={k_value} and similarity_threshold={sim_threshold}")

    # evaluation
    results = evaluate_system(image_embs, text_embs, meta, img_idx, text_idx, k=k_value, similarity_threshold=sim_threshold)

    print("\nFinal Evaluation Results:")
    for mode, metrics in results.items():
        print(f"  {mode.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"    {metric.replace('_', ' ').title()}@{k_value}: {value:.4f}")
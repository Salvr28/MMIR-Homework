import os
import json
import re
from PIL import Image, ImageFile 
import torch
from torch.utils.data import Dataset, get_worker_info 
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor, ProcessorMixin 
import timm 
from timm.data import resolve_data_config, create_transform 

# Usefull for PIL to handle corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Specific UNI parameters, for dummy model to get data configs
UNI_BASE_TIMM_KWARGS_DATA_CONFIG = {
    'img_size': 384,
    'patch_size': 16,
    'in_chans': 3,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'drop_path_rate': 0.1,
    'qkv_bias': True,
    'norm_layer': torch.nn.LayerNorm, 
    'act_layer': torch.nn.GELU,      
    'global_pool': 'avg',
    'init_values': 1e-5,
    'dynamic_img_size': True
}

class ARCHDataset(Dataset):
    def __init__(self, config, target_sets="all"):
        self.config = config
        
        if isinstance(target_sets, str):
            if target_sets == "all":
                self.target_sets = ["pubmed_set", "books_set"]
            else:
                self.target_sets = [target_sets]
        else:
            self.target_sets = target_sets

        image_model_name = config['model']['image_encoder']['name'] 
        text_model_name = config['model']['text_encoder']['name'] 

        # Image processors
        self.image_processor = None 
        self.timm_image_processor = None 

        # UNI is not supported by AutoModel, it has to be configured handcrafted
        if image_model_name == "MahmoodLab/uni":
            # Oad Timm data configuration
            try:
                # Dummy model to get data config
                dummy_model = timm.create_model(
                    f"hf-hub:{image_model_name}", 
                    pretrained=False, 
                    num_classes=0, 
                    **UNI_BASE_TIMM_KWARGS_DATA_CONFIG 
                )
                data_config = resolve_data_config(dummy_model.pretrained_cfg, model=dummy_model, verbose=False)
                self.timm_image_processor = create_transform(**data_config)
                worker_info = get_worker_info()
                if worker_info is None or worker_info.id == 0:
                    print(f"Using timm's create_transform for MahmoodLab/uni.")
            except Exception as e:
                worker_info = get_worker_info()
                if worker_info is None or worker_info.id == 0:
                    print(f"Error loading timm transform for MahmoodLab/uni: {e}. Falling back to Hugging Face AutoFeatureExtractor.")
                self.image_processor = AutoFeatureExtractor.from_pretrained(image_model_name)
        else:
            # For others models we use AutoProcessor
            try:
                self.processor = AutoProcessor.from_pretrained(image_model_name)
                
                if isinstance(self.processor, ProcessorMixin) and hasattr(self.processor, 'image_processor') and hasattr(self.processor, 'tokenizer'):
                    self.image_processor = self.processor.image_processor
                    worker_info = get_worker_info()
                    if worker_info is None or worker_info.id == 0:
                        print(f"Using AutoProcessor (multimodal) for model: {image_model_name}")
                else:
                    self.image_processor = self.processor
                    worker_info = get_worker_info()
                    if worker_info is None or worker_info.id == 0:
                        print(f"Using AutoProcessor (ImageProcessor directly) for {image_model_name}")

            except Exception as e:
                worker_info = get_worker_info()
                if worker_info is None or worker_info.id == 0:
                    print(f"Error with AutoProcessor for {image_model_name}: {e}. Falling back to AutoFeatureExtractor.")
                self.image_processor = AutoFeatureExtractor.from_pretrained(image_model_name)

        # Autotokenizer is the same for all models
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        worker_info = get_worker_info()
        if worker_info is None or worker_info.id == 0:
            print(f"Using AutoTokenizer for text model: {text_model_name}")

        self.text_max_length = config['preprocessing']['text_max_length']
        self.data = self._load_data()
        print(f"Initialized ARCHDataset for {self.target_sets} with {len(self.data)} samples.")

    def _extract_caption_part(self, full_caption, letter):
        """
        Extracts the specific part of a caption corresponding to a given letter.
        Falls back to the full caption if no specific part is found or letter is "Single".
        """
        if not full_caption or not letter or letter == "Single":
            return full_caption

        escaped_letter = re.escape(letter)
        pattern = rf'(?:^|\s|\()?(?:Figure\s*\d+\s*)?(?:{escaped_letter}[,.:\)]|\b{escaped_letter}\b)\s*(.*?)(?=\s*(?:[A-Z][,.:\)]|\b[A-Z]\b|$))'
        
        match = re.search(pattern, full_caption, re.DOTALL | re.IGNORECASE)

        if match:
            extracted_part = match.group(1).strip()
            if extracted_part:
                return extracted_part
        
        return full_caption

    def _get_image_path(self, images_base_path, uuid):
        """
        Constructs and verifies the full path to an image file.
        Checks for both .jpg and .png extensions.

        Args:
            images_base_path (str): The base directory where image files are stored.
            uuid (str): The UUID of the image file (without extension).

        Returns:
            str: The full path to the image file if found and readable, otherwise None.
        """
        for ext in ["jpg", "png"]:
            potential_image_path = os.path.join(images_base_path, f"{uuid}.{ext}")
            if os.path.exists(potential_image_path):
                try:
                    # Test if image is readable
                    test_image = Image.open(potential_image_path).convert("RGB")
                    _ = test_image.size 
                    return potential_image_path
                except Exception:
                    # If image is corrupted return none
                    return None
        return None


    def _load_data(self):
        """
        Loads data from the specified JSON captions files,
        performing robust checks for image and caption validity.
        Only valid samples with existing and readable images are included.
        Handles specific caption extraction for "books_set" using _extract_caption_part.
        """
        processed_data = []
        total_skipped_images_not_found = 0
        total_skipped_corrupted_images = 0 
        total_skipped_no_caption_uuid = 0
        total_skipped_empty_extracted_caption = 0
        total_entries_processed_in_raw_json = 0
        
        for set_name in self.target_sets:
            current_set_path = ""
            if set_name == "pubmed_set":
                current_set_path = self.config['dataset']['pubmed_dir']
            elif set_name == "books_set":
                current_set_path = self.config['dataset']['book_dir']
            else:
                print(f"Warning: Invalid set_name in target_sets: {set_name}. Skipping.")
                continue

            caption_filepath = os.path.join(current_set_path, 'captions.json')
            images_base_path = os.path.join(current_set_path, 'images')

            if not os.path.exists(caption_filepath):
                print(f"Warning: Caption file not found for {set_name} at {caption_filepath}. Skipping this set.")
                continue
            if not os.path.exists(images_base_path):
                print(f"Warning: Images directory not found for {set_name} at {images_base_path}. Skipping this set.")
                continue

            try:
                with open(caption_filepath, 'r') as f:
                    raw_data = json.load(f)
                    total_entries_processed_in_raw_json += len(raw_data)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {caption_filepath}: {e}. Skipping this set.")
                continue

            print(f"Processing {len(raw_data)} entries from {set_name}...")

            for key, item in raw_data.items():
                caption = item.get("caption")
                uuid = item.get("uuid")

                if set_name == "pubmed_set":
                    letter = "Single" 
                else:
                    letter = item.get('letter')


                if not caption or not uuid:
                    total_skipped_no_caption_uuid += 1
                    continue

                # Safe path due to testing
                final_image_path = self._get_image_path(images_base_path, uuid)
                
                if final_image_path is None:
                    if not os.path.exists(os.path.join(images_base_path, f"{uuid}.jpg")) and \
                       not os.path.exists(os.path.join(images_base_path, f"{uuid}.png")):
                        total_skipped_images_not_found += 1
                    else:
                        total_skipped_corrupted_images += 1
                    continue

                if set_name == "books_set":
                    extracted_caption = self._extract_caption_part(caption, letter)
                else:
                    extracted_caption = caption
                
                if not extracted_caption:
                    total_skipped_empty_extracted_caption += 1
                    continue

                processed_data.append({
                    "image_path_full": final_image_path,
                    "text": extracted_caption,
                    "original_full_caption": caption,
                    "uuid": uuid,
                    "letter": letter,
                    "source_set": set_name
                })

        print(f"\n--- Data Loading Summary ---")
        print(f"Total raw JSON entries considered (approx): {total_entries_processed_in_raw_json}")
        print(f"Skipped due to missing caption/uuid: {total_skipped_no_caption_uuid}")
        print(f"Skipped due to image file not found: {total_skipped_images_not_found}")
        print(f"Skipped due to corrupted/unreadable image: {total_skipped_corrupted_images}")
        print(f"Skipped due to empty extracted/final caption: {total_skipped_empty_extracted_caption}")
        print(f"Successfully loaded samples: {len(processed_data)}")
        print(f"----------------------------\n")

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample['image_path_full']
        text = sample['text']
        original_text = sample['original_full_caption']

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 'main'

        try:
            image = Image.open(img_path).convert("RGB")
            
            # Choice for specific processor
            if self.timm_image_processor is not None:
                processed_image = self.timm_image_processor(image) 
            elif self.image_processor is not None:
                processed_image = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            else:
                raise ValueError("No image processor initialized. Check model configuration.")

            # SANITY CHECK 
            if processed_image is None or not isinstance(processed_image, torch.Tensor) or processed_image.numel() == 0:
                raise ValueError(f"Image processing returned invalid tensor for {img_path}")

            encoding = self.tokenizer(
                text,
                max_length=self.text_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            if 'input_ids' not in encoding or encoding['input_ids'] is None or encoding['input_ids'].numel() == 0:
                raise ValueError(f"Tokenizer returned invalid input_ids for text: {text}")
            if 'attention_mask' not in encoding or encoding['attention_mask'] is None or encoding['attention_mask'].numel() == 0:
                raise ValueError(f"Tokenizer returned invalid attention_mask for text: {text}")

            for key in ['original_full_caption', 'image_path_full', 'uuid', 'letter', 'source_set']:
                if sample[key] is None:
                    raise ValueError(f"Metadata key '{key}' is None for UUID {sample.get('uuid', 'N/A')}")

            return {
                'image': processed_image,
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'original_text': original_text,
                'caption': text,
                'image_path': img_path,
                'uuid': sample['uuid'],
                'letter': sample['letter'],
                'source_set': sample['source_set']
            }
        except Exception as e:
            if worker_info is None or worker_info.id == 0:
                print(f"Warning (Worker {worker_id}): Skipping sample with UUID {sample.get('uuid', 'N/A')} at path {img_path} due to specific error: {type(e).__name__}: {e}")
            return None 

import torch
import torch.nn as nn
from transformers import AutoModel
import timm

# UNI specific parameters
UNI_BASE_TIMM_KWARGS = {
    'init_values': 1e-5,
    'dynamic_img_size': True
}

class ImageEncoder(nn.Module):
    """
    An image encoder that loads pre-trained models from Hugging Face. 
    It returns features from the model's backbone.
    """
    def __init__(self, model_name='histai/hibou-b', pretrained=True):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_name)
        
        # Determines the outputdim
        self.output_dim = self.model.config.hidden_size 

    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            features = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            # CLS token is in position 0
            features = output.last_hidden_state[:, 0, :] 
        else:
            raise ValueError("Model output does not contain 'pooler_output' or 'last_hidden_state'. "
                             "Cannot extract features for image encoder.")
        
        return features 

class TimmImageEncoder(nn.Module):
    """
    Specific Image encoder to handle MahmoodLab/uni from timm library.
    """
    def __init__(self, model_name='MahmoodLab/uni', pretrained=True):
        super().__init__()
        self.model_name = model_name
        # Specific model load using timm
        self.model = timm.create_model(f"hf-hub:{model_name}", pretrained=pretrained, **UNI_BASE_TIMM_KWARGS)
        
        # Check output dimension of UNi model
        self.output_dim = self.model.num_features 
        
        print(f"Loaded {model_name} using timm. Output dim: {self.output_dim}")

    def forward(self, pixel_values):
        # Timm models often have a forward_features method that returns token embeddings.
        # For the global embedding, we average the tokens.
        features = self.model.forward_features(pixel_values) # Dimensions (batch_size, num_tokens, embed_dim)
        
        return features.mean(dim=1) # Mean on all tokens


class TextEncoder(nn.Module):
    """
    A text encoder that loads pre-trained models from Hugging Face. 
    It returns features from the model's backbone. 
    """
    def __init__(self, model_name='dmis-lab/biobert-v1.1'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        # Output dimension
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # IF the model use CLS
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            embeddings = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            # Otherwise we built the embedding handcrafted (mean of all tokens)
            last_hidden_state = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            raise ValueError("Model output does not contain 'pooler_output' or 'last_hidden_state'. "
                             "Cannot extract features for text encoder.")
                             
        return embeddings 

class MultimodalModel(nn.Module):
    """
    A multimodal model that integrates an ImageEncoder and a TextEncoder. 
    Its primary purpose is to generate and return normalized embeddings directly from the encoders, 
    without additional projection layers, assuming that the encoders have outputs of the same dimension. 
    It does not include training logic due to technical limitation.
    """
    def __init__(self, config):
        super().__init__()
        
        # Image encoder inizialization
        image_model_name = config['model']['image_encoder'].get('name', 'histai/hibou-b')
        image_pretrained = config['model']['image_encoder'].get('pretrained', True)

        if image_model_name == "MahmoodLab/uni":
            self.image_encoder = TimmImageEncoder(
                model_name=image_model_name,
                pretrained=image_pretrained
            )
        else:
            self.image_encoder = ImageEncoder(
                model_name=image_model_name,
                pretrained=image_pretrained
            )
        
        # Text Encoder inizialization
        text_model_name = config['model']['text_encoder'].get('name', 'nlpie/compact-biobert') 

        self.text_encoder = TextEncoder(
            model_name=text_model_name
        )

        # Check if two encoders have same output dimensions
        if self.image_encoder.output_dim != self.text_encoder.output_dim:
            raise ValueError(
                f"ImageEncoder output dimension ({self.image_encoder.output_dim}) "
                f"does not match TextEncoder output dimension ({self.text_encoder.output_dim}). "
                "All selected encoders must have the same hidden_size (e.g., 768)."
            )
        
        self.embedding_dim = self.image_encoder.output_dim 

    def forward(self, images=None, input_ids=None, attention_mask=None):
        """
        Generates L2-normalized embeddings for images and/or texts.

        Args:
            images (torch.Tensor, optional): Tensor of pre-processed images.
            input_ids (torch.Tensor, optional): Tensor of text token IDs.
            attention_mask (torch.Tensor, optional): Tensor of text attention mask.

        Returns:
            tuple: Contains image embeddings and/or text embeddings.
                Embeddings are normalized for cosine similarity.
                All will have the dimension 'self.embedding_dim'.
        """
        image_embeddings = None
        text_embeddings = None

        if images is not None:
            # Passes images to image encoder to obtains features
            image_embeddings = self.image_encoder(images)
            # Embeddings normalization for cosine similarity
            image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        if input_ids is not None and attention_mask is not None:
            # Passes texts to text encoder to obatin features
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            # normalization
            text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
            
        return image_embeddings, text_embeddings
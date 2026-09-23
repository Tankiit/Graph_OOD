from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseModelWrapper
from .hook_manager import HookManager
from ..utils.device import get_device, to_device


def pool_representations(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling: str = "last"
) -> torch.Tensor:
    """Pool sequence activations into a single representation vector per sample.

    Args:
        hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim].
        attention_mask: Optional binary mask of shape [batch_size, seq_len].
        pooling: One of 'last', 'first', 'cls', 'mean', 'all'.

    Returns:
        Tensor of shape [batch_size, hidden_dim] (or [batch_size, seq_len, hidden_dim] if 'all').
    """
    if pooling == "all" or hidden_states.dim() == 2:
        return hidden_states

    batch_size, seq_len, hidden_dim = hidden_states.shape

    if pooling in ("first", "cls"):
        return hidden_states[:, 0, :]

    if pooling == "last":
        if attention_mask is not None:
            # Find index of last unmasked token per sample
            lengths = attention_mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0, max=seq_len - 1).long()
            batch_idx = torch.arange(batch_size, device=hidden_states.device)
            return hidden_states[batch_idx, lengths, :]
        else:
            return hidden_states[:, -1, :]

    if pooling == "mean":
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        else:
            return hidden_states.mean(dim=1)

    raise ValueError(f"Unknown pooling method '{pooling}'. Choose from 'last', 'first', 'cls', 'mean', 'all'.")


class TransformerWrapper(BaseModelWrapper):
    """Wrapper for Hugging Face Transformers models with layer extraction and steering hooks."""

    def __init__(
        self,
        model_name_or_path: Union[str, nn.Module],
        model_type: str = "causal",
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
        tokenizer: Optional[Any] = None,
        trust_remote_code: bool = False,
    ):
        target_device = get_device(device) if isinstance(device, str) or device is None else device

        if isinstance(model_name_or_path, str):
            if model_type == "causal":
                raw_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype or torch.float32,
                    trust_remote_code=trust_remote_code
                )
            elif model_type == "classification":
                raw_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype or torch.float32,
                    trust_remote_code=trust_remote_code
                )
            else:
                raw_model = AutoModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype or torch.float32,
                    trust_remote_code=trust_remote_code
                )
            raw_model = raw_model.to(target_device)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raw_model = model_name_or_path.to(target_device)
            self.tokenizer = tokenizer

        super().__init__(raw_model)
        self.device = target_device
        self.hook_manager = HookManager(self.model)
        self._layer_names = self._discover_transformer_layers()

    def _discover_transformer_layers(self) -> List[str]:
        """Automatically identify repeating transformer block layer names."""
        candidates = []
        for name, mod in self.model.named_modules():
            # Match standard layer block architectures
            # GPT-2: transformer.h.0
            # LLaMA / Mistral / Qwen: model.layers.0
            # BERT / RoBERTa: bert.encoder.layer.0, roberta.encoder.layer.0
            # DeBERTa: deberta.encoder.layer.0
            parts = name.split(".")
            if len(parts) >= 2 and parts[-1].isdigit():
                parent_name = parts[-2]
                if parent_name in ("h", "layers", "layer", "block", "blocks"):
                    candidates.append(name)

        if not candidates:
            # Fallback: find any modules ending in integer index
            candidates = [name for name, _ in self.model.named_modules() if name.split(".")[-1].isdigit()]

        return candidates

    def get_layer_names(self) -> List[str]:
        return list(self._layer_names)

    def get_layer_module(self, layer_name: str) -> nn.Module:
        return self.hook_manager.get_submodule(layer_name)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def extract_representations(
        self,
        batch: Union[Dict[str, torch.Tensor], List[str], str],
        layer_names: Optional[List[str]] = None,
        pooling: str = "last",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract hidden representations across specified layers.

        Args:
            batch: Dict with 'input_ids' and 'attention_mask', or raw text string/list.
            layer_names: Subset of layer names to extract from (defaults to all).
            pooling: 'last', 'mean', 'cls', 'first', or 'all'.

        Returns:
            Dictionary of layer_name -> Tensor.
        """
        self.model.eval()
        target_layers = layer_names or self.get_layer_names()

        if isinstance(batch, (str, list)):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required when passing raw text strings.")
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=kwargs.get("max_length", 512)
            )
            inputs = to_device(encoded, self.device)
        else:
            inputs = to_device(batch, self.device)

        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            with self.hook_manager.capture_activations(target_layers) as storage:
                _ = self.model(**inputs)

        # Process and pool recorded activations
        representations: Dict[str, torch.Tensor] = {}
        for name in target_layers:
            # storage[name] contains list of activations recorded during forward
            acts = storage[name]
            if len(acts) == 1:
                tensor = acts[0]
            elif len(acts) > 1:
                tensor = torch.cat(acts, dim=0)
            else:
                raise RuntimeError(f"No activations recorded for layer '{name}'")

            representations[name] = pool_representations(tensor, attention_mask=attention_mask, pooling=pooling)

        return representations

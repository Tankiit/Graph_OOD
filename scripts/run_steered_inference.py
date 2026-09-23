#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.utils.device import get_device, to_device
from src.utils.logging import setup_logging, get_logger
from src.models.transformer_wrapper import TransformerWrapper
from src.steering.base import SteeringVector
from src.steering.interveners import AdditiveIntervener

logger = get_logger("SteeredInference")


def main():
    parser = argparse.ArgumentParser(description="Run steered generation and classification inference.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or path.")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Prompt text.")
    parser.add_argument("--steering-vector", type=str, default=None, help="Path to steering vector (.pt).")
    parser.add_argument("--target-layer", type=str, default=None, help="Layer to inject vector into.")
    parser.add_argument("--coefficients", type=float, nargs="+", default=[-2.0, 0.0, 2.0], help="List of coefficients to test.")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max tokens to generate.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Loading model: {args.model} on {device}")
    model = TransformerWrapper(args.model, model_type="causal", device=device)

    if args.steering_vector and Path(args.steering_vector).exists():
        sv = SteeringVector.load(args.steering_vector)
        target_layer = args.target_layer or sv.layer_names[len(sv.layer_names) // 2]
        vector = sv[target_layer]
    else:
        logger.info("No steering vector provided, generating synthetic steering vector for demonstration...")
        target_layer = args.target_layer or model.get_layer_names()[len(model.get_layer_names()) // 2]
        # Create random normalized direction
        torch.manual_seed(args.seed)
        hidden_dim = 768
        for p in model.model.parameters():
            if p.dim() >= 2:
                hidden_dim = p.shape[-1]
                break
        vector = torch.randn(hidden_dim)
        vector = vector / torch.norm(vector, p=2)

    logger.info(f"Target steering layer: {target_layer}")
    logger.info(f"Prompt: '{args.prompt}'\n")

    input_ids = model.tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    print("=" * 70)
    print(f"PROMPT: {args.prompt}")
    print("=" * 70)

    for coeff in args.coefficients:
        intervener = AdditiveIntervener(vector=vector, coefficient=coeff)
        interventions = {target_layer: intervener}

        with model.hook_manager.apply_steering(interventions):
            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=model.tokenizer.eos_token_id
                )

        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[Steering Coeff = {coeff:+.2f}]:\n{generated_text}\n" + "-" * 70)


if __name__ == "__main__":
    main()

"""
ML8502 @MBZUAI
KGW Watermark Exercise
======================

This exercise implements a red-green list watermark for text generation (Kirchenbauer et al.).
The watermark uses a pseudorandom function (PRF) to partition the vocabulary into "green" and
"red" tokens based on the previous token and a secret key. During generation, green tokens
receive a logit boost, making them more likely. Detection measures the green token rate and
computes a z-score to determine if text is watermarked.

Your task: Fill in the two TODO sections to complete the implementation.
Run the script to verify your solution passes all unit tests.
"""

import hashlib
import math
import torch
import numpy as np
from typing import Tuple

# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
SECRET_KEY = 'ml8502-2025'


def prf_greenlist(prev_token_id: int, vocab_size: int, secret_key: str, gamma: float) -> torch.BoolTensor:
    """
    Generate a pseudorandom green list for the given previous token.

    Args:
        prev_token_id: The ID of the previous token in the sequence
        vocab_size: Total size of the vocabulary
        secret_key: Secret key for the PRF (used in hashing)
        gamma: Fraction of vocabulary to mark as green (between 0 and 1)

    Returns:
        Boolean tensor of shape (vocab_size,) where True indicates green tokens
    """
    if not (0.0 < gamma < 1.0):
        raise ValueError('gamma must be in (0, 1)')

    # Create a string combining the secret key and token ID
    material = f'{secret_key}:{prev_token_id}'.encode('utf-8')

    # Hash the material using SHA-256
    digest = hashlib.sha256(material).digest()

    # Convert first 8 bytes to an integer for the RNG seed
    seed = int.from_bytes(digest[:8], 'big', signed=False)

    # Create a random number generator with the seed
    rng = np.random.default_rng(seed)

    # Calculate the number of green tokens
    k = max(1, math.ceil(gamma * vocab_size))

    # Randomly select k indices without replacement
    indices = rng.choice(vocab_size, size=k, replace=False)

    # Create boolean mask
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[indices] = True

    return mask

# =============================================================================
# TODO 1: Implement watermarked text generation
# =============================================================================

def generate_with_watermark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    secret_key: str,
    gamma: float = 0.5,
    delta: float = 2.0,
    device: torch.device = DEVICE
) -> Tuple[str, torch.Tensor]:
    """
    Generate text with watermark by boosting green token logits.

    Args:
        model: Causal language model (e.g., Qwen2.5)
        tokenizer: Tokenizer for the model
        prompt: Initial text prompt
        max_new_tokens: Number of tokens to generate
        secret_key: Secret key for watermark PRF
        gamma: Fraction of vocabulary to mark as green
        delta: Logit boost for green tokens
        device: Device to run generation on

    Returns:
        Tuple of (generated_text: str, token_ids: torch.Tensor)
    """
    model.eval()
    encoded = tokenizer(prompt, return_tensors='pt').to(device)
    generated = encoded.input_ids

    # Use model's actual vocab size to avoid indexing issues
    vocab_size = model.config.vocab_size

    for _ in range(max_new_tokens):
        # Get model outputs for current sequence
        outputs = model(input_ids=generated)

        # Extract logits for the last position (next token prediction)
        logits = outputs.logits[:, -1, :]  # Shape: (batch_size=1, vocab_size)

        # Get the previous token ID (last token in the sequence so far)
        prev_token = int(generated[0, -1])

        # ToDo: Fill in token selection here
        # <FILL NEXT TOKEN SELECTION HERE>

        raise NotImplementedError("Complete TODO: Implement generate_with_watermark")

        # final statement should append the next token to the list 'generated'
        # generated = torch.cat([generated, next_token], dim=1)

    # Decode and return
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, generated[0].detach().cpu()


# =============================================================================
# TODO 2: Implement watermark detection
# =============================================================================

def detect_watermark(token_ids: torch.Tensor, vocab_size: int, secret_key: str, gamma: float) -> Tuple[float, float]:
    """
    Detect watermark by computing green token rate and z-score.

    Args:
        token_ids: Tensor of token IDs in the sequence
        vocab_size: Size of the vocabulary
        secret_key: Secret key used for watermarking
        gamma: Green list fraction (must match generation)

    Returns:
        Tuple of (green_rate: float, z_score: float)

    Hints:
        green_rate = hits / total
        z-score: (green_rate - gamma) / sqrt(gamma * (1-gamma) / total)
    """
    tokens = token_ids.tolist()
    hits = 0
    total = 0

    ## ToDo: Count hits and total tokens
    raise NotImplementedError("Complete TODO: Implement detect_watermark")

# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    """Load Qwen2.5-0.5B model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ Error: transformers not installed")
        print("   Run: pip install transformers")
        return None, None

    print(f"Loading Qwen2.5-0.5B on {DEVICE}...")

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
        device_map=DEVICE
    )
    model.eval()

    print(f"✓ Model loaded (vocab_size={tokenizer.vocab_size})")
    return model, tokenizer


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run watermark tests with Qwen2.5-0.5B."""
    print("=" * 70)
    print("KGW Watermark Exercise - Testing with Qwen2.5-0.5B")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        return 1

    # Use model's actual vocab size (handles special tokens correctly)
    vocab_size = model.config.vocab_size

    try:
        # Test parameters
        gamma = 0.4
        delta = 2.0
        prompt = "The future of artificial intelligence"

        print(f"\n📝 Prompt: '{prompt}'")
        print(f"⚙️  Parameters: gamma={gamma}, delta={delta}, vocab_size={vocab_size}")

        # Generate watermarked text
        print("\n" + "=" * 70)
        print("Test 1: Watermarked Generation")
        print("=" * 70)

        watermarked_text, watermarked_tokens = generate_with_watermark(
            model, tokenizer, prompt,
            max_new_tokens=150,
            secret_key=SECRET_KEY,
            gamma=gamma,
            delta=delta,
            device=DEVICE
        )

        print(f"\n🔒 Watermarked text:\n{watermarked_text}")

        # Detect watermark
        wm_green_rate, wm_z_score = detect_watermark(
            watermarked_tokens, vocab_size, SECRET_KEY, gamma
        )
        print(f"\n📊 Detection results:")
        print(f"   Green rate: {wm_green_rate:.3f} (expected: {gamma:.3f})")
        print(f"   Z-score: {wm_z_score:.2f}")

        # Generate unwatermarked baseline
        print("\n" + "=" * 70)
        print("Test 2: Unwatermarked Baseline (delta=0)")
        print("=" * 70)

        unwatermarked_text, unwatermarked_tokens = generate_with_watermark(
            model, tokenizer, prompt,
            max_new_tokens=150,
            secret_key=SECRET_KEY,
            gamma=gamma,
            delta=0.0,  # No watermark
            device=DEVICE
        )

        print(f"\n🔓 Unwatermarked text:\n{unwatermarked_text}")

        # Detect on unwatermarked
        unwm_green_rate, unwm_z_score = detect_watermark(
            unwatermarked_tokens, vocab_size, SECRET_KEY, gamma
        )
        print(f"\n📊 Detection results:")
        print(f"   Green rate: {unwm_green_rate:.3f} (expected: {gamma:.3f})")
        print(f"   Z-score: {unwm_z_score:.2f}")

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Watermarked z-score:   {wm_z_score:+.2f}")
        print(f"Unwatermarked z-score: {unwm_z_score:+.2f}")
        print(f"Difference:            {wm_z_score - unwm_z_score:+.2f}")

        # Validation
        if wm_z_score > 2.0:
            print("\n✅ Watermark detected! (z-score > 2.0)")
        else:
            print(f"\n⚠️  Watermark weak (z-score = {wm_z_score:.2f})")

        if wm_z_score > unwm_z_score:
            print("✅ Watermarked text has higher z-score than unwatermarked")
            print("\n" + "=" * 70)
            print("ALL TESTS PASSED! ✓")
            print("=" * 70)
            return 0
        else:
            print("❌ Watermarked text does not have higher z-score")
            return 1

    except NotImplementedError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease complete the TODO sections and try again.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

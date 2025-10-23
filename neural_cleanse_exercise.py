"""
ML8502 @MBZUAI
Neural Cleanse Exercise - Backdoor Forensics
=============================================

Your task: Complete the neural_cleanse() optimization loop to reverse-engineer backdoor triggers.

Background:
- You have a backdoored ResNet-50 model trained on ImageNet
- Neural Cleanse (Wang et al., 2019) finds backdoors by optimizing triggers for each class
- For clean classes: requires large, complex triggers → large mask norm
- For backdoored classes: recovers the actual backdoor → small mask norm

Your Mission:
1. Complete the TODO section in the neural_cleanse() function (around line 253)
2. Run the script to analyze the model
3. Identify which class was poisoned (look for anomalous mask norm)
4. Describe what the recovered trigger looks like
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from typing import Tuple, List
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"Running on device: {DEVICE}")


# =============================================================================
# Dataset Loader
# =============================================================================

class ImageNetSampleDataset(Dataset):
    """Load images from imagenet_sample folder."""

    def __init__(self, image_dir: str = 'imagenet_sample'):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob('*.JPEG')))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        print(f"✓ Loaded {len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, 0  # Return dummy label


def load_data(batch_size: int = 32):
    """Load ImageNet sample dataset."""
    dataset = ImageNetSampleDataset('imagenet_sample')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


# =============================================================================
# Model Loader
# =============================================================================

def load_backdoored_model(checkpoint_path: str = 'backdoored_resnet50.pth'):
    """Load the backdoored ResNet-50 model."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Backdoored model not found at {checkpoint}")

    print(f"Loading backdoored model from {checkpoint}...")

    # Create ResNet-50 architecture
    model = models.resnet50(weights=None)

    # Load backdoored weights
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("✓ Model loaded successfully")
    return model


# =============================================================================
# Helper Functions
# =============================================================================

def project_trigger_and_mask(trigger: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project trigger and mask to valid ranges.

    The trigger is clamped to valid normalized pixel values.
    The mask is clamped to [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN, device=trigger.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=trigger.device).view(1, 3, 1, 1)

    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std

    trigger.data = torch.max(torch.min(trigger.data, max_val), min_val)
    mask.data.clamp_(0.0, 1.0)

    return trigger, mask


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


# =============================================================================
# Helper Function: Trigger Application (PROVIDED - DO NOT MODIFY)
# =============================================================================

def apply_trigger(images: torch.Tensor, trigger: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Blend trigger into a batch of images using the soft mask.

    Args:
        images: Batch of images (N, C, H, W) in normalized space
        trigger: Trigger pattern (1, C, H, W) in normalized space
        mask: Soft blending mask (1, 1, H, W) with values in [0, 1]

    Returns:
        Blended images (N, C, H, W)

    Formula:
        blended = images * (1 - mask) + trigger * mask
    """
    if images.ndim != 4:
        raise ValueError('images must be NCHW format')

    # Expand trigger to match batch size if needed
    trigger_batch = trigger
    if trigger_batch.size(0) == 1:
        trigger_batch = trigger_batch.expand(images.size(0), -1, -1, -1)

    # Expand mask to match batch size
    mask_batch = mask
    if mask_batch.size(0) == 1:
        mask_batch = mask_batch.expand(images.size(0), -1, -1, -1)

    # Expand mask channels to match image channels
    if mask_batch.size(1) == 1:
        mask_batch = mask_batch.expand(-1, images.size(1), -1, -1)

    # Apply soft blending: images * (1 - mask) + trigger * mask
    blended = images * (1 - mask_batch) + trigger_batch * mask_batch

    return blended


# =============================================================================
# YOUR TASK: Complete the Neural Cleanse optimization loop
# =============================================================================

def neural_cleanse(
    model: nn.Module,
    data_loader: DataLoader,
    suspected_label: int,
    steps: int = 250,
    lr: float = 0.1,
    lambda_l1: float = 0.0001,
    device: torch.device = DEVICE,
    batches_per_step: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Optimize a trigger and mask that induces the suspected target label.

    Goal: Find the SMALLEST trigger (measured by mask L1 norm) that can make
          the model predict suspected_label for any input image.

    Args:
        model: Pre-trained image classifier
        data_loader: DataLoader providing clean images
        suspected_label: Target class to trigger (e.g., 42 for tree_frog)
        steps: Number of optimization steps
        lr: Learning rate for Adam optimizer
        lambda_l1: Weight for L1 mask regularization (encourages small masks)
        device: Device to run optimization on
        batches_per_step: Number of batches to accumulate per step (for stability)

    Returns:
        Tuple of (optimized_trigger, optimized_mask, loss_history)

    Algorithm:
        1. Initialize trigger and mask as learnable parameters
        2. For each optimization step:
           a. Zero gradients
           b. Accumulate loss over batches_per_step batches:
              - Apply trigger to clean images
              - Compute cross-entropy loss to encourage predicted_label = suspected_label
              - Add L1 regularization on mask to encourage sparsity
           c. Backpropagate accumulated loss
           d. Update parameters with optimizer.step()
           e. Project trigger and mask to valid ranges
           f. Record loss
        3. Return optimized trigger, mask, and loss history
    """
    model.eval()  # Keep model in evaluation mode

    # Initialize trigger and mask as learnable parameters
    # Trigger: Start with small random noise (will be optimized)
    # Mask: Start very small (will grow to necessary size during optimization)
    trigger = torch.randn(1, 3, 224, 224, device=device) * 0.1
    trigger.requires_grad = True
    mask = torch.full((1, 1, 224, 224), 0.01, device=device)
    mask.requires_grad = True

    # Create Adam optimizer for trigger and mask (PROVIDED)
    optimizer = torch.optim.Adam([trigger, mask], lr=lr)

    # Create infinite iterator over data loader
    data_iter = itertools.cycle(data_loader)
    loss_history: List[float] = []

    # Main optimization loop
    for step in range(steps):
        # Zero gradients (PROVIDED)
        optimizer.zero_grad()
        total_loss = 0.0

        # Accumulate gradient over multiple batches for stability
        for _ in range(batches_per_step):
            # Get a batch of clean images
            images, _ = next(data_iter)
            images = images.to(device)

            # ============================================================
            # TODO: Implement the Neural Cleanse optimization
            # ============================================================
            raise NotImplementedError("neural cleanse needs to be implemented")

        # Backpropagate and update parameters (PROVIDED)
        total_loss.backward()
        optimizer.step()

        # Project trigger and mask to valid ranges (PROVIDED)
        project_trigger_and_mask(trigger, mask)

        # Record loss for analysis
        loss_history.append(total_loss.item())

    # Return optimized trigger and mask (detached from computation graph)
    return trigger.detach(), mask.detach(), loss_history


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(results: dict, class_names: dict, top_n: int = 10):
    """Visualize Neural Cleanse results."""

    # Sort classes by mask norm
    mask_norms = {cls_idx: results[cls_idx]['mask_norm'] for cls_idx in results}
    sorted_classes = sorted(mask_norms.items(), key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("Mask Norms by Class (sorted)")
    print("=" * 70)
    print(f"{'Class':<6} {'Class Name':<30} {'Mask Norm':<12} {'Anomaly'}")
    print("-" * 70)

    # Find outlier (significantly smaller than median)
    norms = sorted([v for v in mask_norms.values()])
    median_norm = norms[len(norms) // 2]
    threshold = median_norm * 0.7  # 30% smaller than median

    for i, (cls_idx, norm) in enumerate(sorted_classes[:top_n]):
        is_anomaly = "⚠️  SUSPICIOUS!" if norm < threshold else ""
        class_name = class_names.get(cls_idx, f"Class {cls_idx}")
        print(f"{cls_idx:<6} {class_name:<30} {norm:<12.6f} {is_anomaly}")

    # Identify the most suspicious class
    most_suspicious = sorted_classes[0][0]
    print("\n" + "=" * 70)
    print(f"Most Suspicious Class: {most_suspicious} ({class_names.get(most_suspicious, 'Unknown')})")
    print(f"Mask Norm: {sorted_classes[0][1]:.6f}")
    print("=" * 70)

    # Visualize the recovered trigger and mask
    trigger = results[most_suspicious]['trigger'].cpu()
    mask = results[most_suspicious]['mask'].cpu()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot trigger
    trigger_img = denormalize(trigger.squeeze(0)).permute(1, 2, 0).clamp(0, 1).numpy()
    axes[0].imshow(trigger_img)
    axes[0].set_title(f'Recovered Trigger Pattern\nClass {most_suspicious} ({class_names.get(most_suspicious, "Unknown")})',
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot mask
    mask_img = mask.squeeze(0).squeeze(0).numpy()
    im = axes[1].imshow(mask_img, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Recovered Mask (L1 norm: {sorted_classes[0][1]:.6f})',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('neural_cleanse_trigger.png', dpi=200, bbox_inches='tight')
    print("\n✓ Trigger visualization saved to neural_cleanse_trigger.png")
    plt.show()

    return most_suspicious


# =============================================================================
# Main Execution
# =============================================================================

# Mapping of class indices to names (subset of ImageNet classes)
CLASS_NAMES = {
    2: "great_white_shark",
    17: "jay",
    42: "tree_frog",
    52: "green_snake"
}


def main():
    """Run Neural Cleanse investigation on backdoored model."""
    print("=" * 70)
    print("Neural Cleanse Exercise - Backdoor Forensics")
    print("=" * 70)

    try:
        # Load model and data
        model = load_backdoored_model('backdoored_resnet50.pth')
        data_loader = load_data(batch_size=16)

        # Classes to investigate
        suspected_classes = [k for k,v in CLASS_NAMES.items()]  # First 20 ImageNet classes

        print(f"\n🔍 Investigating {len(suspected_classes)} classes...")
        print("⚙️  Running Neural Cleanse with 100 optimization steps per class")
        print("⏱️  This will take a few minutes...\n")

        results = {}

        # Run Neural Cleanse for each suspected class
        for i, idx in enumerate(suspected_classes, 1):
            class_name = CLASS_NAMES.get(idx, f"Class {idx}")
            print(f"[{i}/{len(suspected_classes)}] Testing {class_name} (class {idx})...", end=' ')

            trigger, mask, history = neural_cleanse(
                model, data_loader,
                suspected_label=idx,
                steps=100,
                lr=0.1,
                lambda_l1=0.0001,
                device=DEVICE,
                batches_per_step=5
            )

            mask_norm = mask.abs().mean().item()
            results[idx] = {
                'trigger': trigger,
                'mask': mask,
                'history': history,
                'mask_norm': mask_norm
            }

            print(f"Mask norm: {mask_norm:.6f}")

        # Visualize and identify the poisoned class
        print("\n" + "=" * 70)
        print("Analysis Complete - Identifying Backdoor")
        print("=" * 70)

        poisoned_class = visualize_results(results, CLASS_NAMES, top_n=20)

        # Summary
        print("\n" + "=" * 70)
        print("🎯 YOUR MISSION")
        print("=" * 70)
        print("1. ❓ Which class was poisoned?")
        print(f"   → Answer: Class {poisoned_class} ({CLASS_NAMES.get(poisoned_class, 'Unknown')})")
        print("\n2. 🔍 What does the trigger look like?")
        print("   → Examine the visualization saved to neural_cleanse_trigger.png")
        print("   → Describe the trigger pattern (shape, location, color)")
        print("\n3. 📊 Why is this class suspicious?")
        print("   → Its mask norm is significantly smaller than others")
        print("   → Smaller mask = smaller trigger = likely backdoor!")
        print("=" * 70)

        return 0

    except NotImplementedError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease complete the TODO sections and try again.")
        return 1
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  - backdoored_resnet50.pth in the current directory")
        print("  - imagenet_sample/ folder with JPEG images")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

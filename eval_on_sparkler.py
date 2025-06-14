import argparse
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sparkler import sparkler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "-s",
        "--sparkler",
        action="store_true",
    )
    args = parser.parse_args()

    # Paths
    IMAGENET_VAL_DIR = "/Users/bryansong/Data/ImageNet2012_val"

    # Load processor and model
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        use_fast=True,
    )
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    if args.sparkler:
        print("✨ Sparkler compression running")
        with torch.no_grad():  # Disable gradient tracking
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(sparkler(param.data).to(param.device))

    # Dataset and preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert to [0,1]
            transforms.Normalize(
                mean=image_processor.image_mean, std=image_processor.image_std
            ),
        ]
    )

    # dataset = ImageFolder(IMAGENET_VAL_DIR, transform=transform)
    from torchvision.datasets import ImageNet

    dataset = ImageNet(root=IMAGENET_VAL_DIR, split="val", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Inference
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(model.device, dtype=torch.float16)
            outputs = model(pixel_values=images)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
            # print("correct/total = ", correct, '/', total, '=', correct/total)

    print(f"Top-1 Accuracy on ImageNet validation set: {correct / total:.4f}")

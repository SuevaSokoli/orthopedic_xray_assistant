import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMAGE_SIZE, BODY_PARTS, CONDITIONS
from src.model import build_model
from src.dataset import IDX_TO_BODY_PART, IDX_TO_CONDITION

# ── Load and preprocess image ──────────────────────────────
def preprocess_image(image_path):
    """
    Loads an image and prepares it for the model.
    Returns both the tensor and the original RGB image.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image_resized) / 255.0  # normalize to [0,1]
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    return image_tensor, image_np

# ── Generate Grad-CAM heatmap ──────────────────────────────
def generate_gradcam(model, image_tensor, image_np, target="condition", device="cpu"):
    """
    Generates a Grad-CAM heatmap for the given image.
    
    Args:
        model: the trained OrthoResNet50 model
        image_tensor: preprocessed image tensor (1, 3, 224, 224)
        image_np: original image as numpy array (224, 224, 3) normalized to [0,1]
        target: "condition" or "body_part" — which head to explain
        device: "cuda" or "cpu"
    
    Returns:
        cam_image: heatmap overlaid on the original image (numpy array)
        body_part_idx: predicted body part index
        condition_idx: predicted condition index
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # Target the last convolutional layer of ResNet50
    target_layer = model.feature_extractor[-1][-1].conv3

    # Define which output to explain
    if target == "condition":
        def target_fn(output):
            body_logits, cond_logits = output
            return cond_logits
    else:
        def target_fn(output):
            body_logits, cond_logits = output
            return body_logits

    # Wrap model to return single output for GradCAM
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, target):
            super().__init__()
            self.model = model
            self.target = target

        def forward(self, x):
            body_logits, cond_logits = self.model(x)
            if self.target == "condition":
                return cond_logits
            return body_logits

    wrapped_model = ModelWrapper(model, target).to(device)
    target_layer = wrapped_model.model.feature_extractor[-1][-1].conv3

    # Generate Grad-CAM
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)
    grayscale_cam = grayscale_cam[0]

    # Overlay heatmap on image
    cam_image = show_cam_on_image(
        image_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # Get predictions
    with torch.no_grad():
        body_logits, cond_logits = model(image_tensor)
        body_part_idx = body_logits.argmax(1).item()
        condition_idx = cond_logits.argmax(1).item()

    return cam_image, body_part_idx, condition_idx


# ── Save Grad-CAM image ────────────────────────────────────
def save_gradcam(cam_image, output_path):
    """Saves the Grad-CAM heatmap to disk."""
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, cam_image_bgr)
    print(f"✅ Grad-CAM saved to: {output_path}")


# ── Full pipeline ──────────────────────────────────────────
def analyze_xray(image_path, model_path, output_path=None, device="cpu"):
    """
    Full analysis pipeline:
    1. Load model
    2. Preprocess image
    3. Generate predictions
    4. Generate Grad-CAM
    5. Return results

    Args:
        image_path: path to the X-ray image
        model_path: path to the saved model checkpoint
        output_path: where to save the Grad-CAM image (optional)
        device: "cuda" or "cpu"

    Returns:
        dict with predictions and Grad-CAM image
    """
    # Load model
    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Preprocess image
    image_tensor, image_np = preprocess_image(image_path)

    # Generate Grad-CAM
    cam_image, body_part_idx, condition_idx = generate_gradcam(
        model, image_tensor, image_np,
        target="condition", device=device
    )

    # Save if output path provided
    if output_path:
        save_gradcam(cam_image, output_path)

    return {
        "body_part": IDX_TO_BODY_PART[body_part_idx],
        "condition": IDX_TO_CONDITION[condition_idx],
        "body_part_idx": body_part_idx,
        "condition_idx": condition_idx,
        "gradcam_image": cam_image
    }
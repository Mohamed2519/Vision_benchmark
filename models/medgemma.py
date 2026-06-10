from abc import ABC, abstractmethod
from typing import List
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from models.base import ChestXrayModel
from benchmarks.registry import ModelRegistry



@ModelRegistry.register("MedGemma")
class MedGemmaModel(ChestXrayModel):
    
    PROMPT = (
        "You are an expert radiologist analyzing a chest X-ray. "
        "Based solely on the visible radiological findings, respond with exactly two values separated by a comma: "
        "First value: 0 if the patient is NORMAL (no significant pathological findings), "
        "or 1 if the patient is ABNORMAL (any disease or abnormality is present). "
        "Second value: your confidence score as a decimal between 0.0 and 1.0. "
        "Output format: label,confidence — for example: 0,0.9 or 1,0.75. "
        "Do not include any other text, explanation, or punctuation."
    )

    def __init__(self, device: str = "auto", model_name: str = "medgemma"):
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.processor = None

    def _load_model_and_processor(self):
        """Load the model and processor."""
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _predict_single(self, image_path: str) -> dict:
        """Make prediction for a single image."""
        image = self.load_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.PROMPT},
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=2000, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        label, confidence = decoded.split(",")
        label = int(label.strip())
        confidence = float(confidence.strip())
        
        return {"label": label, "score": confidence}

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """Return [{"score": float, "label": int}, ...] for each image path."""
        # Load model if not already loaded
        if self.model is None or self.processor is None:
            self._load_model_and_processor()
        
        results = []
        for image_path in image_paths:
            try:
                result = self._predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({"label": -1, "score": 0.0})  # Error indicator
        
        return results


if __name__ == "__main__":
    # Example usage
    image_paths = [
        r"/mnt/nvme/echonova-vision/dev/pathology_binary_classifier/dataset/vinDr/golden_images_vinDr/0a61578e3d77b1cebc86d13a41efa31b.png",
        # Add more paths as needed
    ]
    
    model = MedGemmaModel(device="auto", model_name="medgemma")
    results = model.predict_batch(image_paths)
    
    for img_path, result in zip(image_paths, results):
        print(f"Image: {img_path}")
        print(f"Label: {result['label']}, Confidence: {result['score']}\n")
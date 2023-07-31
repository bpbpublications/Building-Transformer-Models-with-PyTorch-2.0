import io
import torch
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import numpy as np

# Initialize the tokenizer and model
id2label={0: 'background', 1: 'candy', 2: 'egg tart', 3: 'french fries', 4: 'chocolate', 5: 'biscuit', 6: 'popcorn', 7: 'pudding', 8: 'ice cream', 9: 'cheese butter', 10: 'cake', 11: 'wine', 12: 'milkshake', 13: 'coffee', 14: 'juice', 15: 'milk', 16: 'tea', 17: 'almond', 18: 'red beans', 19: 'cashew', 20: 'dried cranberries', 21: 'soy', 22: 'walnut', 23: 'peanut', 24: 'egg', 25: 'apple', 26: 'date', 27: 'apricot', 28: 'avocado', 29: 'banana', 30: 'strawberry', 31: 'cherry', 32: 'blueberry', 33: 'raspberry', 34: 'mango', 35: 'olives', 36: 'peach', 37: 'lemon', 38: 'pear', 39: 'fig', 40: 'pineapple', 41: 'grape', 42: 'kiwi', 43: 'melon', 44: 'orange', 45: 'watermelon', 46: 'steak', 47: 'pork', 48: 'chicken duck', 49: 'sausage', 50: 'fried meat', 51: 'lamb', 52: 'sauce', 53: 'crab', 54: 'fish', 55: 'shellfish', 56: 'shrimp', 57: 'soup', 58: 'bread', 59: 'corn', 60: 'hamburg', 61: 'pizza', 62: ' hanamaki baozi', 63: 'wonton dumplings', 64: 'pasta', 65: 'noodles', 66: 'rice', 67: 'pie', 68: 'tofu', 69: 'eggplant', 70: 'potato', 71: 'garlic', 72: 'cauliflower', 73: 'tomato', 74: 'kelp', 75: 'seaweed', 76: 'spring onion', 77: 'rape', 78: 'ginger', 79: 'okra', 80: 'lettuce', 81: 'pumpkin', 82: 'cucumber', 83: 'white radish', 84: 'carrot', 85: 'asparagus', 86: 'bamboo shoots', 87: 'broccoli', 88: 'celery stick', 89: 'cilantro mint', 90: 'snow peas', 91: ' cabbage', 92: 'bean sprouts', 93: 'onion', 94: 'pepper', 95: 'green beans', 96: 'French beans', 97: 'king oyster mushroom', 98: 'shiitake', 99: 'enoki mushroom', 100: 'oyster mushroom', 101: 'white button mushroom', 102: 'salad', 103: 'other ingredients'}
label2id={'background': 0, 'candy': 1, 'egg tart': 2, 'french fries': 3, 'chocolate': 4, 'biscuit': 5, 'popcorn': 6, 'pudding': 7, 'ice cream': 8, 'cheese butter': 9, 'cake': 10, 'wine': 11, 'milkshake': 12, 'coffee': 13, 'juice': 14, 'milk': 15, 'tea': 16, 'almond': 17, 'red beans': 18, 'cashew': 19, 'dried cranberries': 20, 'soy': 21, 'walnut': 22, 'peanut': 23, 'egg': 24, 'apple': 25, 'date': 26, 'apricot': 27, 'avocado': 28, 'banana': 29, 'strawberry': 30, 'cherry': 31, 'blueberry': 32, 'raspberry': 33, 'mango': 34, 'olives': 35, 'peach': 36, 'lemon': 37, 'pear': 38, 'fig': 39, 'pineapple': 40, 'grape': 41, 'kiwi': 42, 'melon': 43, 'orange': 44, 'watermelon': 45, 'steak': 46, 'pork': 47, 'chicken duck': 48, 'sausage': 49, 'fried meat': 50, 'lamb': 51, 'sauce': 52, 'crab': 53, 'fish': 54, 'shellfish': 55, 'shrimp': 56, 'soup': 57, 'bread': 58, 'corn': 59, 'hamburg': 60, 'pizza': 61, ' hanamaki baozi': 62, 'wonton dumplings': 63, 'pasta': 64, 'noodles': 65, 'rice': 66, 'pie': 67, 'tofu': 68, 'eggplant': 69, 'potato': 70, 'garlic': 71, 'cauliflower': 72, 'tomato': 73, 'kelp': 74, 'seaweed': 75, 'spring onion': 76, 'rape': 77, 'ginger': 78, 'okra': 79, 'lettuce': 80, 'pumpkin': 81, 'cucumber': 82, 'white radish': 83, 'carrot': 84, 'asparagus': 85, 'bamboo shoots': 86, 'broccoli': 87, 'celery stick': 88, 'cilantro mint': 89, 'snow peas': 90, ' cabbage': 91, 'bean sprouts': 92, 'onion': 93, 'pepper': 94, 'green beans': 95, 'French beans': 96, 'king oyster mushroom': 97, 'shiitake': 98, 'enoki mushroom': 99, 'oyster mushroom': 100, 'white button mushroom': 101, 'salad': 102, 'other ingredients': 103}
feature_extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(
    "prem-timsina/segformer-b0-finetuned-food",
    id2label=id2label,
    label2id=label2id
)
# Initialize FastAPI
app = FastAPI()

@app.post("/segment/")
async def segment_image(file: UploadFile):
    # Open image and convert to RGB
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Transform the image
    input_tensor = feature_extractor(images=[image], return_tensors="pt")

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the input tensor to the correct device
    input_tensor = input_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**input_tensor)
        predictions = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()

    # The output is a tensor with the segmented image
    # For simplicity, we will convert it to a binary mask (0s and 1s)
    grayscale_map = np.zeros((predictions.shape[0], predictions.shape[1]), dtype=np.uint8)
    for label_id in id2label.keys():
        grayscale_map[predictions == label_id] = label_id

    # Convert the grayscale map to a PIL image
    segmentation_image = Image.fromarray(grayscale_map, mode='L')

    # Create a dictionary for the counts of each class in the image
    unique, counts = np.unique(predictions, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # Map the class IDs back to their names
    class_counts = {id2label[k]: v for k, v in class_counts.items()}

    # Create a new image to put the text onto
    annotated_image = Image.new('RGB', (segmentation_image.width, segmentation_image.height + len(class_counts)*10), 'white')

    # Paste the original image onto the new one
    annotated_image.paste(segmentation_image, (0,0))

    # Create a draw object
    draw = ImageDraw.Draw(annotated_image)

    # Draw the class_counts onto the new image
    for i, class_name in enumerate(class_counts.keys()):
        draw.text((0, segmentation_image.height + i*10), class_name, fill='black')

    # Save the annotated image as a png file
    annotated_image.save("annotated_mask.png")

    # Return the png file
    return StreamingResponse(io.BytesIO(open("annotated_mask.png", "rb").read()), media_type="image/png")

import os
from flask import Flask, request, jsonify
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from google.generativeai import configure, GenerativeModel 

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

#Configure Gemini API
GEMINI_API_KEY = "AIzaSyCVQHI_ArRIWqOmqDwi0D1cC5kBKXq-gVI"
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel("gemini-pro")


# Initialize Flask app
app = Flask(__name__)

@app.route("/caption", methods=["POST"])
def generate_caption():

    if "image" not in request.files:
        return jsonify({"error": "No image found in request"}), 400

    image_file = request.files["image"]

    # Save the uploaded image temporarily
    temp_path = "temp.jpg"
    image_file.save(temp_path)

    raw_image = Image.open(temp_path).convert("RGB")

    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Clean up temp file
    os.remove(temp_path)

    return jsonify({"caption": caption})

def extract_keyword(sentence):
    """Extract the most relevant keyword from a sentence using Gemini API."""
    prompt = f"Extract the most important keyword from this sentence: '{sentence}'. Only return the keyword."
    response = gemini_model.generate_content(prompt)
    return response.text.strip().lower()

@app.route("/filter-images", methods=["POST"])
def filter_images():
    if "images" not in request.files or "sentence" not in request.form:
        return jsonify({"error": "Images and sentence are required"}), 400

    images = request.files.getlist("images") # Array of images
    sentence = request.form["sentence"] # Input sentence

    keyword = extract_keyword(sentence)

    matching_images = []

    for image_file in images:
        # Save temporarily & open image
        temp_path = f"temp_{image_file.filename}"
        image_file.save(temp_path)
        raw_image = Image.open(temp_path).convert("RGB")

        # Generate caption
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"Image: {image_file.filename} | Caption: {caption}")  # Debugging log

        # Check if the keyword is in the caption
        if keyword in caption.lower():
            matching_images.append({"filename": image_file.filename, "caption": caption})

        # Cleanup
        os.remove(temp_path)

    return jsonify({"keyword": keyword, "matching_images": matching_images})

# Run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

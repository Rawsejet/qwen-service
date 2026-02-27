#!/bin/bash

# Process all images in a folder using Qwen3-VL-4B
# Sends all images in a single request

python3 << 'PYTHON_EOF'
import json
import requests
import glob
import sys

# Update this path to your images folder
images_folder = "/home/teja/Documents/pics/screenshots"

# Get all image files
images = sorted(glob.glob(f"{images_folder}/*")[:10])  # Limit to first 10 images

print(f"Found {len(images)} images:")
for img in images:
    print(f"  - {img}")

# Build message with text prompt followed by all images
content = [{"type": "text", "text": "These are a set of screenshots that contain a coding problem. Extract just the coding problem text and ignore all other parts on the screen."}]

for img in images:
    content.append({"type": "image_url", "image_url": {"url": f"file://{img}"}})

payload = {
    "model": "Qwen3-VL-4B",
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 1024
}

response = requests.post("http://127.0.0.1:8088/v1/chat/completions", json=payload)
result = response.json()

# Print just the assistant's response
if "choices" in result and len(result["choices"]) > 0:
    print("\n=== Response ===")
    print(result["choices"][0]["message"]["content"])
PYTHON_EOF

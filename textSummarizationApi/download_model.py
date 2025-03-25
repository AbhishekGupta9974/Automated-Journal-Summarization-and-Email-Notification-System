import gdown
import os

# Google Drive file ID
file_id = "1Xm30rvW-bFQ9mj1dvfCj3b971yomC_qL" 
output_path = "final_model/model.safetensors"

# Create the directory if not exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

print("Model downloaded successfully!")

import os
import sys
from datasets import load_dataset
from PIL import Image
import io

class Tee:
    def __init__(self, file_name):
        self.file = open(file_name, "a")  # Open file in append mode
        self.console = sys.stdout  # Store the original console output

    def write(self, message):
        # Write to both the console and the file
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        # Ensure the message is written to both destinations
        self.console.flush()
        self.file.flush()

    def close(self):
        # Close the file when done
        self.file.close()

 # Redirect all print statements to Tee
tee = Tee("huggingface_dataset_loader_for_validation_output.txt")
sys.stdout = tee

# Get the HF_HOME environment variable
hf_home = os.getenv("HF_HOME")
if not hf_home:
    print("HF_HOME environment variable is required, but not set.")
    sys.exit(1)

# Get the CAP6640_PROJECT_DIR environment variable
cap6640_project_dir = os.getenv("CAP6640_PROJECT_DIR")
if not cap6640_project_dir:
    print("CAP6640_PROJECT_DIR environment variable is required, but not set.")
    sys.exit(1)

# Load the required dataset (JamieWithofs/Deepfake-and-real-images) from Hugging Face, for the validation set
dataset = load_dataset("JamieWithofs/Deepfake-and-real-images", split="validation", cache_dir=hf_home)

# Get the number of entries in the dataset
number_of_entries_in_dataset = len(dataset)

# Create directories for saving images, based on labels
os.makedirs(f"{cap6640_project_dir}/Images/HuggingFace/JamieWithofs/Deepfake-and-real-images/validation", exist_ok=True)

# Get the unique labals for each entry in the dataset
labels = set(entry["label"] for entry in dataset)
for label in labels:
    # Create label subdirectories
    os.makedirs(f"{cap6640_project_dir}/Images/HuggingFace/JamieWithofs/Deepfake-and-real-images/validation/{label}", exist_ok=True)

# Get, process and save the images
print("Getting the images from the Hugging Face dataset, processing & saving them locally...\n")

for idx, entry in enumerate(dataset):
    
    # Print which image is being processed
    print(f"{(idx+1)}/{number_of_entries_in_dataset}: {entry}")

    # Load the image as a PIL Image object
    image = entry["image"]  # PIL Image object
    byte_io = io.BytesIO() # In-memory buffer to store binary data
    image.save(byte_io, format="JPEG")  # Saves the PIL image as a byte stream of the JPEG/PNG image
    byte_io.seek(0)  # Rewind to the beginning of the byte stream in the buffer

    # Get the unique label for this entry
    label = entry["label"]

    # Read the image in the byte format
    image_bytes = byte_io.read()

    # Save the image in the appropriate directory based on its label
    with open(f"{cap6640_project_dir}/Images/HuggingFace/JamieWithofs/Deepfake-and-real-images/validation/{label}/image_{idx}.jpg", "wb") as f:
        f.write(image_bytes)
    
    # Break after processing a subset
    # if idx == 2:
    #    break

# Reset stdout to the default (console)
sys.stdout = sys.__stdout__

import os
import sys
from datasets import load_dataset
import numpy as np
import requests.adapters
import soundfile as sf
import io
import requests

class Tee:
    def __init__(self, file_name):
        self.file = open(file_name, "w")  # Open file in append mode
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
tee = Tee("huggingface_audio_dataset_loader_for_train_output.txt")
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

# Set HTTP(S) request retries
requests.adapters.DEFAULT_RETRIES = 5  # Set retry attempts
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))

# Load the required dataset (Hemg/Deepfakeaudio) from Hugging Face, for the train set
dataset = load_dataset("Hemg/Deepfakeaudio", split="train", cache_dir=hf_home)

# Get the number of entries in the dataset
number_of_entries_in_dataset = len(dataset)

# Create directories for saving audio, based on labels
os.makedirs(f"{cap6640_project_dir}/Audio/HuggingFace/Hemg/Deepfakeaudio/train", exist_ok=True)

# Get the unique labals for each entry in the dataset
labels = set(entry["label"] for entry in dataset)
for label in labels:
    # Create label subdirectories
    os.makedirs(f"{cap6640_project_dir}/Audio/HuggingFace/Hemg/Deepfakeaudio/train/{label}", exist_ok=True)

print()

# Get, process and save the audio
print("Getting the audio from the Hugging Face dataset, processing & saving them locally...\n")

for idx, entry in enumerate(dataset):
    
    # Print which audio is being processed
    print(f"{(idx+1)}/{number_of_entries_in_dataset}: {entry}")

    # Get the raw waveform data and its sampling rate
    audio_array = entry["audio"]["array"]
    sampling_rate = entry["audio"]["sampling_rate"]

    # Get the unique label for this entry
    label = entry["label"]

    # Generate a file name including the full path to save the wavefrom to
    # The waveform is saved as a .wav file
    file_name = os.path.join(f"{cap6640_project_dir}/Audio/HuggingFace/Hemg/Deepfakeaudio/train/{label}", f"audio_{idx}.wav")

    # Write to the .wav file
    sf.write(file_name, audio_array, sampling_rate)
    
    # # Break after processing a subset
    # if idx == 2:
    #    break

# Reset stdout to the default (console)
sys.stdout = sys.__stdout__

import os
import sys
from openai import OpenAI
import base64
import json
from pathlib import Path
import random
from datetime import datetime
import time

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
tee = Tee("openapi_deepfake_audio_detector_for_test_1000_output.txt")
sys.stdout = tee

# Get the OPENAI_API_KEY environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("API Key not found")
    sys.exit(1)

# Get the CAP6640_PROJECT_DIR environment variable
cap6640_project_dir = os.getenv("CAP6640_PROJECT_DIR")
if not cap6640_project_dir:
    print("CAP6640_PROJECT_DIR environment variable is required, but not set.")
    sys.exit(1)

client = OpenAI()

# Set the root directory for the audio, get all the filepaths in the root directory, and shuffle the list randomly
audio_root_dir = Path(f"{cap6640_project_dir}/Audio/HuggingFace/Hemg/Deepfakeaudio/train")
file_paths = list(audio_root_dir.rglob('*'))
random.shuffle(file_paths)

# Get the audio count
total_audio_count = sum(1 for filepath in file_paths if filepath.is_file())

# Current audio being processed
current_audio_number = 0

# Initialize this variable to "yes", which means the audio is deepfake
deepfake_detected = "yes"

# Initialize variables for the computation of Precision, Recall, F1 Score and AUC Curve
# for the Confusion Matrix
num_true_positive = 0
num_false_poitive = 0
num_false_negative = 0
num_true_negative = 0

# Get the current time
start_time = time.time()

print("Beginning processing audio...")
print()

# Iterate through all files in the root directory and subdirectories
for filepath in file_paths:
    # Check if it is a file and not a directory
    if filepath.is_file():
        current_audio_number += 1 # Increment the audio number being processed
        audio_path = filepath.as_posix() # Replace '\' (Windows-style) to '/' (Linux-style)
        with open(audio_path, "rb") as audio_file:
            audio_data_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

        split_audio_path = audio_path.split('/')

        # Check if there are at least two parts to the audio path after splitting
        if len(split_audio_path) >= 2:
            # Get the second last character, which represents whether the audio is deepfake or genuine
            second_last_part = split_audio_path[-2]
            last_part = split_audio_path[-1]
        else:
            print("The audio path does not have enough parts to split.")
            sys.exit(1)

        # Send the audio to OpenAPI API for analysis
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio= {"voice": "alloy", "format": "wav"},
            messages=[
                {"role": "system", "content": "You are an audio forensic expert that can detect a deepfake audio."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tell me if there are synthesis artifacts, altered sounds, inconsistencies in speech, background noise, imperfect expression of emotions, robotic-sounding speech, unnatural pauses, or errors in the audio encoding. If you can detect any of these, you must return yes, else you must return no."},
                        {"type": "input_audio", "input_audio": {"data": audio_data_base64, "format": "wav"}}
                    ],
                }
            ],
        )

        # # Convert the response object to a dictionary and then pretty-print
        # response_dict = response.to_dict()

        # # Pretty-print the response
        # print("Response:")
        # # print(response_dict)
        # print(json.dumps(response_dict, indent=4, sort_keys=True))

        # True Positive - the audio is deepfake and is detected deepfake
        if (second_last_part == "0") and (deepfake_detected.lower() in response.choices[0].message.audio.transcript.lower()):
            num_true_positive += 1
        # False Positive - the audio is not deepfake but is detected deepfake
        elif (second_last_part == "1") and (deepfake_detected.lower() in response.choices[0].message.audio.transcript.lower()):
            num_false_poitive += 1
        # False Negative - the audio is deepfake but is not detected deepfake
        elif (second_last_part == "0") and not (deepfake_detected.lower() in response.choices[0].message.audio.transcript.lower()):
            num_false_negative += 1
        # True Negative - the audio is not deepfake and is not detected deepfake
        else:
            num_true_negative += 1

        label = 'Deepfake' if second_last_part == "0" else 'Not Deepfake'

        # Get the current local time
        current_time = datetime.now()

        print(f"{current_time.strftime("%Y-%m-%d %H:%M:%S"):<25} {current_audio_number:<10} / {total_audio_count:<10} {last_part:<25} Label? {label:<20} Deepfake detected? {response.choices[0].message.audio.transcript}")

        # Introduce a delay of 1 second before making another request to OpenAPI, to avoid a Rate Limit issue
        # time.sleep(1)

        # Break after processing a subset
        if current_audio_number == 1000:
            break

# Get the current time
end_time = time.time()

print()

print(f"Time taken to complete processing {current_audio_number} audio file: {(end_time - start_time):.0f} seconds")

print()

print(f"Number of True Positives:  {num_true_positive}")
print(f"Number of False Positives: {num_false_poitive}")
print(f"Number of False Negatives: {num_false_negative}")
print(f"Number of True Negatives:  {num_true_negative}")

print()

try:
    accuracy = ((num_true_positive + num_true_negative) / (num_true_positive + num_true_negative + num_false_poitive + num_false_negative)) * 100
except ZeroDivisionError:
    print("Cannot calculate Accuracy")
    accuracy = 0

try:
    precision = ((num_true_positive) / (num_true_positive + num_false_poitive)) * 100
except ZeroDivisionError:
    print("Cannot calculate Precision")
    precision = 0

try:
    recall = ((num_true_positive) / (num_true_positive + num_false_negative)) * 100
except ZeroDivisionError:
    print("Cannot calculate Recall")
    recall = 0

try:
    f1_score = (2 * (precision) * (recall)) / (precision + recall)
except ZeroDivisionError:
    print("Cannot calculate F1 Score")
    f1_score = 0

print()

print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall:    {recall:.2f}%")
print(f"F1 Score:  {f1_score:.2f}%")

print()
print("-" * 50)
print()
print()

# Reset stdout to the default (console)
sys.stdout = sys.__stdout__

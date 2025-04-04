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
tee = Tee("openapi_deepfake_image_detector_for_train_and_test_1000_output.txt")
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

# Convert image to base64 encoded string


def image_to_base64(image_path):
    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Read the image file and encode it to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


# Set the root directory for the train images, get all the filepaths in the root directory, and shuffle the list randomly
image_root_dir_train = Path(
    f"{cap6640_project_dir}/Images/HuggingFace/JamieWithofs/Deepfake-and-real-images/train")
file_paths_train = [file_path for file_path in image_root_dir_train.rglob(
    '*') if file_path.is_file()]
random.shuffle(file_paths_train)
top_5_file_paths_train = file_paths_train[:5]

# Get the image count
total_image_count_train = sum(
    1 for filepath in file_paths_train if filepath.is_file())

# Current image being processed
current_image_number_train = 0

print()
print("Beginning adding training data...")
print()

message = []

for filepath_train in top_5_file_paths_train:
    current_image_number_train += 1  # Increment the image number being processed
    image_path_train = filepath_train.as_posix()
    image_typ = "image/jpeg"  # Type of image
    image_as_base64_encoded_string_train = image_to_base64(
        image_path_train)  # Convert image to base64 encoded string

    split_image_path_train = image_path_train.split('/')

    # Check if there are at least two parts to the image path after splitting
    if len(split_image_path_train) >= 2:
        # Get the second last character, which represents whether the image is deepfake or genuine
        second_last_part_train = split_image_path_train[-2]
        last_part_train = split_image_path_train[-1]
    else:
        print("The image path does not have enough parts to split.")
        sys.exit(1)

    label_train = 'yes' if second_last_part_train == "0" else 'no'

    message.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me if synthesis artifacts are in the image. You must return with yes or no only."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_typ};base64,{image_as_base64_encoded_string_train}"},
                },
            ],
        },
    )

    message.append(
        {"role": "assistant", "content": f"{label_train}"},
    )

    # Get the current local time
    current_time = datetime.now()

    print(f"{current_time.strftime("%Y-%m-%d %H:%M:%S"):<25} {current_image_number_train:<10} / {total_image_count_train:<10} {last_part_train:<25} Label? {label_train:<20}")

print()

# Set the root directory for the test images, get all the filepaths in the root directory, and shuffle the list randomly
image_root_dir_test = Path(
    f"{cap6640_project_dir}/Images/HuggingFace/JamieWithofs/Deepfake-and-real-images/test")
file_paths_test = list(image_root_dir_test.rglob('*'))
random.shuffle(file_paths_test)

# Get the image count
total_image_count_test = sum(
    1 for filepath in file_paths_test if filepath.is_file())

# Current image being processed
current_image_number_test = 0

# Initialize this variable to "Yes" if the LLM detects that the image is deepfake
deepfake_detected = "Yes"

# Initialize variables for the computation of Precision, Recall, F1 Score and AUC Curve
# for the Confusion Matrix
num_true_positive = 0
num_false_poitive = 0
num_false_negative = 0
num_true_negative = 0

# Get the current time
start_time = time.time()

print()
print("Beginning processing images...")
print()

# Iterate through all files in the root directory and subdirectories
for filepath_test in file_paths_test:
    # Check if it is a file an not a directory
    if filepath_test.is_file():
        current_image_number_test += 1  # Increment the image number being processed
        # Replace '\' (Windows-style) to '/' (Linux-style)
        image_path_test = filepath_test.as_posix()
        image_typ = "image/jpeg"  # Type of image
        image_as_base64_encoded_string_test = image_to_base64(
            image_path_test)  # Convert image to base64 encoded string

        split_image_path_test = image_path_test.split('/')

        # Check if there are at least two parts to the image path after splitting
        if len(split_image_path_test) >= 2:
            # Get the second last character, which represents whether the image is deepfake or genuine
            second_last_part_test = split_image_path_test[-2]
            last_part_test = split_image_path_test[-1]
        else:
            print("The image path does not have enough parts to split.")
            sys.exit(1)

        messages = [
            {"role": "system", "content": "You are an image forensic expert that can detect a deepfake image."},
        ]

        messages.extend(message)
        
        messages.append(
            {
                "role": "user",
                "content": [
                        {"type": "text",
                            "text": "Tell me if synthesis artifacts are in the image. You must return with 1) yes or no only; 2) if yes, explain where the artifacts exist by answering in [region, artifacts] form."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_typ};base64,{image_as_base64_encoded_string_test}"},
                        },
                ],
            },
        )

        # Send the image to OpenAPI API for analysis
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        # # Convert the response object to a dictionary and then pretty-print
        # response_dict = response.to_dict()

        # # Pretty-print the response
        # print("Response:")
        # print(response_dict)
        # print(json.dumps(response_dict, indent=4, sort_keys=True))

        # True Positive - the image is deepfake and is detected deepfake
        if (second_last_part_test == "0") and (deepfake_detected.lower() in response.choices[0].message.content.lower()):
            num_true_positive += 1
        # False Positive - the image is not deepfake but is detected deepfake
        elif (second_last_part_test == "1") and (deepfake_detected.lower() in response.choices[0].message.content.lower()):
            num_false_poitive += 1
        # False Negative - the image is deepfake but is not detected deepfake
        elif (second_last_part_test == "0") and not (deepfake_detected.lower() in response.choices[0].message.content.lower()):
            num_false_negative += 1
        # True Negative - the image is not deepfake and is not detected deepfake
        else:
            num_true_negative += 1

        label_test = 'Deepfake' if second_last_part_test == "0" else 'Not Deepfake'

        # Get the current local time
        current_time = datetime.now()

        print(
            f"{current_time.strftime("%Y-%m-%d %H:%M:%S"):<25} {current_image_number_test:<10} / {total_image_count_test:<10} {last_part_test:<25} Label? {label_test:<20} Deepfake detected? {response.choices[0].message.content}")

        # Introduce a delay of 15 seconds before making another request to OpenAI API, to avoid a Rate Limit issue
        time.sleep(15)

        # Break after processing a subset
        if current_image_number_test == 1000:
            break

# Get the current time
end_time = time.time()

print()

print(
    f"Time taken to complete processing {current_image_number_test} images: {(end_time - start_time):.0f} seconds")

print()

print(f"Number of True Positives:  {num_true_positive}")
print(f"Number of False Positives: {num_false_poitive}")
print(f"Number of False Negatives: {num_false_negative}")
print(f"Number of True Negatives:  {num_true_negative}")

accuracy = ((num_true_positive + num_true_negative) / (num_true_positive +
            num_true_negative + num_false_poitive + num_false_negative)) * 100
precision = ((num_true_positive) /
             (num_true_positive + num_false_poitive)) * 100
recall = ((num_true_positive) / (num_true_positive + num_false_negative)) * 100
f1_score = (2 * (precision) * (recall)) / (precision + recall)

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

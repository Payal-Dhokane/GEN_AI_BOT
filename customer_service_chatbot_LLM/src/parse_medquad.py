import os
import csv
import xml.etree.ElementTree as ET

# Path to MedQuAD dataset
DATASET_PATH = "../MedQuAD-master/MedQuAD-master"

# Output CSV
OUTPUT_CSV = "../dataset/medical_dataset.csv"

rows = []

# Traverse all folders
for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".xml"):

            file_path = os.path.join(root, file)

            try:

                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                # Disease / Topic
                focus = root_xml.findtext("Focus", default="Unknown")

                # All QA pairs
                for qa in root_xml.findall(".//QAPair"):

                    question = qa.findtext("Question", default="").strip()

                    answer = qa.findtext("Answer", default="").strip()

                    if question and answer:

                        rows.append([
                            focus,
                            question,
                            answer
                        ])

            except Exception as e:

                print(f"Error reading {file}: {e}")

# Save CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "focus",
        "prompt",
        "response"
    ])

    writer.writerows(rows)

print(f"\nDone!")
print(f"Total Question-Answer pairs: {len(rows)}")
print(f"Saved to: {OUTPUT_CSV}")
# from LanguagePDF_Model import process_single_document
from language_model import process_single_document
import os


def process_multiple_documents(directory_path, model_type, user_prompt):
    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

    # Ensure there are PDF files in the directory
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the specified directory.")

    results = {}

    # Iterate through each PDF file and process it individually
    for pdf_file in pdf_files:
        local_path = os.path.join(directory_path, pdf_file)
        print(f"Processing file: {local_path}")
        result = process_single_document(local_path, model_type, user_prompt)
        results[pdf_file] = result

    return results


# directory_path = "/Users/admin/Documents/pdf_file_blobs"
# model_type = "llama3"
# user_prompt = input("Enter prompt here: ")

# results = process_multiple_documents(directory_path, model_type, user_prompt)

# # Print results for each PDF document
# for pdf_file, result in results.items():
#     print(f"Results for {directory_path}/{pdf_file}:\n{result}\n")
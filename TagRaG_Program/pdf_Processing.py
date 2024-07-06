import os
import pandas as pd
from langchain_community.llms import Ollama
from process_Model import process_multiple_documents

def process_pdfs(directory_path, user_prompt, model_type, data_types, word_count):
    cached_llm = Ollama(model=model_type)
    results = process_multiple_documents(directory_path, model_type, user_prompt)

    file_paths = []
    responses = []
    prompts = []

    for pdf_file, result in results.items():
        file_paths.append(os.path.join(directory_path, pdf_file))
        responses.append(result)
        prompts.append(user_prompt)

    res_df = pd.DataFrame({
        "file_path": file_paths,
        "summary": responses,
        "prompt": prompts,
    })

    if len(data_types) == 0:
        return res_df
    else:
        tags = []
        for _, row in res_df.iterrows():
            text = row["summary"]

            if word_count == 1:
                prompt = f"""Classify the following text based on the given tags:

Text: {text}

Available Tags: {', '.join(data_types)}

Please assign the most suitable tag to this text. Respond with only the tag."""

                response = cached_llm.invoke(prompt)
                tag = response.strip().split()[0]
                tags.append(tag)

            elif word_count == 2:
                prompt = f"""Classify the following text based on the given tags:

Text: {text}

Available Tags: {', '.join(data_types)}

Please assign the most suitable two tags to this text. Respond with only the tags, separated by a comma."""

                response = cached_llm.invoke(prompt)
                tag_list = response.strip().split(',')
                tag_list = [tag.strip() for tag in tag_list[:2]]
                tags.append(', '.join(tag_list))

        res_df["tags"] = tags
        return res_df

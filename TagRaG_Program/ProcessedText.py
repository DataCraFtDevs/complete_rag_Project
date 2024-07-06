import os
import pandas as pd
from langchain_community.llms import Ollama
from LanguageText_Model import processLanguageModel_Text

def process_dataframe(doc_path, model_type, user_prompt, word_count, data_types):
    # Initialize the model
    cached_llm = Ollama(model=model_type)
    
    # Load the CSV file
    df_data = pd.read_csv(doc_path)
    
    # Process each text and classify
    dataRes = []
    prompt = df_data.shape[0] * user_prompt
    
    # Process each text and store results
    for index, text in enumerate(df_data["Text"].values):
        result = processLanguageModel_Text(text, model_type, user_prompt, index)
        dataRes.append(result)
    
    # Add results and prompt to DataFrame
    df_data["Prompt_Results"] = dataRes
    df_data["prompt"] = user_prompt
    
    # If no data_types are provided, print the DataFrame
    if len(data_types) == 0:
        print(df_data)
    else:
        tags = []
        for text in df_data["prompt_results"].values:
            if word_count == 1:
                classification_prompt = f"""Classify the following text based on the given tags:
                
                Text: {text}
                
                Available Tags: {', '.join(data_types)}
                
                Please assign the most suitable tag to this text. Respond with only the tag."""
    
                response = cached_llm.invoke(classification_prompt)
                
                # Extract the first word from the response, assuming it's the tag
                tag = response.strip().split()[0]
                tags.append(tag)
    
            elif word_count == 2:
                classification_prompt = f"""Classify the following text based on the given tags:
                
                Text: {text}
                
                Available Tags: {', '.join(data_types)}
                
                Please assign the most suitable two tags to this text. Respond with only the tags, separated by a comma."""
    
                response = cached_llm.invoke(classification_prompt)
                
                # Extract the tags from the response, assuming they're separated by a comma
                tag_list = response.strip().split(',')
                tag_list = [tag.strip() for tag in tag_list[:2]]  # Ensure we only take the required number of tags
                
                tags.append(', '.join(tag_list)) 
            
        # Add the tags to the DataFrame
        df_data["tags"] = tags
        
        # Print the final DataFrame
        return df_data
    
        # Save the final DataFrame to a CSV file
        # df_data.to_csv("processed_data.csv", index=False)

# # Example usage
# doc_path = "/Users/admin/Documents/pdf_file_blobs/New_Data_TextDummy_data.csv"
# model_type = "llama3"
# user_prompt = "summarize this poem with less than 40 words"
# word_count = 2
# data_types = ['finance', 'government', 'banking', 'agriculture', 'people management', 'education', 'economics', 'legal', 'love']

# process_dataframe(doc_path, model_type, user_prompt, word_count, data_types)

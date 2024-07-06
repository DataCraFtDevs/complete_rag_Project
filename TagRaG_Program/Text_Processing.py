import pandas as pd
from langchain_community.llms import Ollama
from LanguageText_Model import processLanguageModel_Text

def process_dataframe(doc_path, model_type, user_prompt, word_count, data_types):
    cached_llm = Ollama(model=model_type)
    df_data = pd.read_csv(doc_path)
    
    dataRes = []
    prompt = df_data.shape[0] * user_prompt

    for index, text in enumerate(df_data["Text"].values):
        result = processLanguageModel_Text(text, model_type, user_prompt, index)
        dataRes.append(result)

    df_data["prompt_results"] = dataRes
    df_data["prompt"] = user_prompt

    if len(data_types) == 0:
        return df_data
    else:
        tags = []
        for text in df_data["prompt_results"].values:
            if word_count == 1:
                classification_prompt = f"""Classify the following text based on the given tags:
                
Text: {text}
                
Available Tags: {', '.join(data_types)}
                
Please assign the most suitable tag to this text. Respond with only the tag."""

                response = cached_llm.invoke(classification_prompt)
                tag = response.strip().split()[0]
                tags.append(tag)

            elif word_count == 2:
                classification_prompt = f"""Classify the following text based on the given tags:
                
Text: {text}
                
Available Tags: {', '.join(data_types)}
                
Please assign the most suitable two tags to this text. Respond with only the tags, separated by a comma."""

                response = cached_llm.invoke(classification_prompt)
                tag_list = response.strip().split(',')
                tag_list = [tag.strip() for tag in tag_list[:2]]
                tags.append(', '.join(tag_list))

        df_data["tags"] = tags
        return df_data

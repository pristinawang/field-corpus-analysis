import os, json
from openai import OpenAI
import pandas as pd
import torch
import torch.nn.functional as F
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Get the parent directory (where corpora_analysis is located)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# print(parent_dir)
# Add it to sys.path
sys.path.append(parent_dir)
# Print sys.path to confirm
# print("Updated sys.path:")
# print("\n".join(sys.path))
from corpora_analysis.metrics import *

def getallarticles(dir_path):
    '''s
    return list type, all gen frames from all articles in this directory
    return list, return ids of all articles that have their frames generated
    ex input: dir_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/immigration"
    '''
    allarticles=[]
    gen_articleids=[]
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):  # Ensure it's a JSON file
            file_path = os.path.join(dir_path, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Extract frames and add them to the combined dictionary
                for article_id, annotations in data.items():
                    allarticles.append(annotations['Text'])
                    gen_articleids.append(article_id)
    
    return allarticles, gen_articleids

def read_input_segments(info,info_exp, data_path="/data/afield6/afield6/merging_input_media_Llama3.18b_selSeg/"):
    '''
    Example for data_path: data_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/"
    return: dict type; key: article id, value: article text
    '''
    if info: print('---------Info: '+str(info)+'-------------')
    areas=["immigration", 'samesex', 'tobacco']
    allarticles_bigls=[]
    allarticleids_bigls=[]
    for area in areas:
        dir_path=data_path+area
        allarticles_ls, gen_articleids_ls=getallarticles(dir_path=dir_path)
        allarticles_bigls = allarticles_bigls+allarticles_ls
        allarticleids_bigls = allarticleids_bigls+gen_articleids_ls
        if info:
            print(area+' num of articles:', len(gen_articleids_ls))
    if info: 
        print('All num of articles:', len(allarticleids_bigls))   


    article_dict = dict(zip(allarticleids_bigls, allarticles_bigls))
    if info_exp: 
        print('----------Example of Dataset(first 3 items from immigration)------------')
        for k,v in dict(list(article_dict.items())[:3]).items(): print(k,v)
    return article_dict

def txt_to_string(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def generate_responses_batch(prompts):
    """
    Takes a list of prompts and returns a list of generated responses.
    
    Args:
        prompts (list): A list of prompt strings.
    
    Returns:
        list: A list of generated responses corresponding to each prompt.
    """
    client = OpenAI()

    responses = []
    
    for prompt in prompts:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        responses.append(completion.choices[0].message.content)  # Extract response text

    return responses

def generate_response_single(prompt):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def save_codes_csv(func_name, iter_num, codebook_dict, file_full_path):
    '''
    iter_num: int
    codes: list of str
    '''
    codes=list(codebook_dict.keys())
    #codes=[code+' {'+code_max_sim_dict[code][0]+', '+str(code_max_sim_dict[code][1])+'}'+' ('+str(len(segs))+')' for code, segs in codebook_dict.items()]
    codes=[code+' ('+str(len(segs))+')' for code, segs in codebook_dict.items()]
    
    column_name = f"{str(iter_num)}_{func_name}"  # Generate column name

    if os.path.exists(file_full_path):  # Check if file exists
        df = pd.read_csv(file_full_path)  # Read existing file
    else:
        df = pd.DataFrame()  # Create empty DataFrame if file doesn't exist

    # Ensure the DataFrame has the correct number of rows
    max_length = max(len(df), len(codes))  # Ensure new column fits
    df = df.reindex(range(max_length))  # Resize if needed
    codes = codes + [None] * (max_length - len(codes))
    df[column_name] = codes  # Add new column

    df.to_csv(file_full_path, index=False)  # Save file
    
def metrics_to_txt(func_name, iter_num,metrics_bool, embedding_model, gold_frame_article_id_dict, hat_frame_article_id_dict, file_full_path):
    if metrics_bool:
        metrics=Metrics(embedding_model=embedding_model)
        metrics.calculate_article_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict, hat_frame_article_id_dict=hat_frame_article_id_dict)
        with open(file_full_path, "a", encoding="utf-8") as f:
            txt=str(func_name)+'_'+str(iter_num)+'; Article metrics: '+str(metrics.article_metric)
            f.write(txt+"\n")  # Writing segment and article ID

def metrics_to_csv(func_name, iter_num,metrics_bool, embedding_model, gold_frame_article_id_dict, hat_frame_article_id_dict, gold_frame_dict, hat_frame_dict, gold_silhouette_score, article_ids_seen, file_full_path):
    if not(metrics_bool): return

    metrics=Metrics(embedding_model=embedding_model)
    metrics_dict=metrics.calculate_all_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict, hat_frame_article_id_dict=hat_frame_article_id_dict, hat_frame_dict=hat_frame_dict, gold_frame_dict=gold_frame_dict, article_ids_seen=article_ids_seen)
    
    

    # Construct the row ID
    row_id = f"{func_name}_{iter_num}"

    # Metrics to log 
    metrics_to_log = {
        "frame_level_precision": metrics_dict["frame_level_precision"],
        "frame_level_recall": metrics_dict["frame_level_recall"],
        "frame_level_f1": metrics_dict["frame_level_f1"],
        "segment_level_precision": metrics_dict["segment_level_precision"],
        "segment_level_recall": metrics_dict["segment_level_recall"],
        "segment_level_f1": metrics_dict["segment_level_f1"],
        "hat_silhouette_score": metrics_dict["hat_silhouette_score"],
        "gold_silhouette_score": gold_silhouette_score,
    }

    # Add the ID as the first column
    row_data = {"id": row_id}
    row_data.update(metrics_to_log)

    # Convert to single-row DataFrame
    new_row_df = pd.DataFrame([row_data])

    # Define expected columns (including "id" as first)
    expected_columns = ["id"] + list(metrics_to_log.keys())

    # Handle file creation or appending
    if os.path.exists(file_full_path):
        existing_df = pd.read_csv(file_full_path)

        if list(existing_df.columns) != expected_columns:
            raise ValueError("Column names in the existing CSV do not match the expected format.")

        new_row_df.to_csv(file_full_path, mode='a', header=False, index=False)
    else:
        new_row_df.to_csv(file_full_path, mode='w', header=True, index=False)

    print(f"Logged metrics for {row_id} to {file_full_path}")

def get_max_label_sim(embedding_model, labels, device='cuda', round_decimal=2):
    '''
    input: 
    labels: list of strings (labels)

    output:
    max_sim_results: dictionary where:
        - keys are labels from the input list
        - values are tuples (most similar label, max cosine similarity score)
    '''

    # Get embeddings for all labels
    label_embeddings = embedding_model.encode(labels, convert_to_tensor=True, device=device)

    # Normalize embeddings for cosine similarity
    label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    # Compute cosine similarity between all labels
    cos_sim_matrix = torch.matmul(label_embeddings, label_embeddings.T)  # Shape: (len(labels), len(labels))

    # Set diagonal to -inf to exclude self-comparison
    cos_sim_matrix.fill_diagonal_(-float('inf'))

    # Find the most similar label for each label
    max_indices = torch.argmax(cos_sim_matrix, dim=1)  # Index of max similarity
    max_sim_scores = torch.max(cos_sim_matrix, dim=1).values  # Max similarity values

    # Convert results to dictionary format
    max_sim_results = {
        labels[i]: (labels[max_indices[i].item()], round(max_sim_scores[i].item(),round_decimal))  # (Most similar label, score)
        for i in range(len(labels))
    }

    return max_sim_results

def plot_stats(dir_path, csv_path):
    
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    df.set_index('id', inplace=True)

    # Prepare shared x-ticks
    x_labels = df.index
    x_pos = range(len(df))

    # Plot and save frame-level precision & recall
    plt.figure(figsize=(14, 6))
    ax = df[['frame_level_precision', 'frame_level_recall']].plot(marker='o', ax=plt.gca())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    plt.title('Frame-level Precision & Recall')
    plt.ylabel('Score')
    plt.xlabel('id')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'frame_precision_recall.png'))
    plt.close()

    # Plot and save segment-level precision & recall
    plt.figure(figsize=(14, 6))
    ax = df[['segment_level_precision', 'segment_level_recall']].plot(marker='o', ax=plt.gca())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    plt.title('Segment-level Precision & Recall')
    plt.ylabel('Score')
    plt.xlabel('id')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'segment_precision_recall.png'))
    plt.close()

    # Plot and save silhouette scores
    plt.figure(figsize=(14, 6))
    ax = df[['hat_silhouette_score', 'gold_silhouette_score']].plot(marker='o', ax=plt.gca())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    plt.title('Silhouette Scores (HAT vs Gold)')
    plt.ylabel('Score')
    plt.xlabel('id')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'silhouette_scores.png'))
    plt.close()

    




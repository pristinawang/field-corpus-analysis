import json
import os
from datasets import Dataset
import torch
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pickle
from goldlabels import *
import random
import string
from datetime import datetime
from metrics import *
from collections import Counter
from get_iter_codebook import get_iter_codebook

def getallframes(dir_path, printerrorframes=False):
    '''
    return list type, all gen frames from all articles in this directory
    return list, return ids of all articles that have their frames generated
    ex input: dir_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/immigration"
    '''
    allframes=[] #all frames from all articles
    gen_articleids=[]
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):  # Ensure it's a JSON file
            file_path = os.path.join(dir_path, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Extract frames and add them to the combined dictionary
                for article_id, annotations in data.items():
                    for annotation in annotations['LLM_Annotation']:
                        if 'frame' in annotation:
                            if not(isinstance(annotation['frame'], str)) or annotation['frame']=='':
                                if printerrorframes: print(filename,article_id, annotation['frame'])
                            else:
                                allframes.append(annotation['frame'])
                        elif 'label' in annotation:
                            for label in annotation['label']:
                                if not(isinstance(label, str)) or label=='':
                                    if printerrorframes: print(filename,article_id, label)
                                else:
                                    allframes.append(label)
                    gen_articleids.append(article_id)
    
    return allframes, gen_articleids

def tokenization(example, tokenizer):
    return tokenizer(example["frame"])

def getembeddings(embedding_model, dataset):
    # use gpu
    # Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    
    # Load Pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(embedding_model)
    model = BertModel.from_pretrained(embedding_model)
    
    # Data prep for model 3steps: tokenization -> data padding with data collator -> dataloader
    tokenized_dataset = dataset.map(lambda x: tokenization(x, tokenizer), batched=True)
    print(tokenized_dataset)
    print(tokenized_dataset[0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    full_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=2, collate_fn=data_collator)
    for batch in full_dataloader:
        print(batch)
        break
    
def get_media_frame_data(data_path,info,csv):
    '''
    Example for data_path: data_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/"
    csv: boolean; to create a csv for lloom, with cols: doc_id, frames(all not unique)
    '''
    if info: print('---------Info: '+str(info)+'-------------')
    areas=["immigration", 'samesex', 'tobacco']
    alluniqueframes_ls=[]
    allframes_bigls=[]
    allarticleids_bigls=[]
    for area in areas:
        dir_path=data_path+area
        allframes_ls, gen_articleids_ls=getallframes(dir_path=dir_path, printerrorframes=False)
        allframes_bigls = allframes_ls+allframes_bigls
        allarticleids_bigls = gen_articleids_ls+allarticleids_bigls
        if info:
            print(area+' unique frames:',len(set(allframes_ls)),'; num of articles:', len(gen_articleids_ls))
    if info: 
        print('All unique frames:', len(set(allframes_bigls)), '; All num of articles:', len(allarticleids_bigls))   
        print('----------------------')
    if csv:
        doc_id=range(1,len(allframes_bigls)+1)
        df = pd.DataFrame({'doc_id': doc_id, 'frames': allframes_bigls})
        df.to_csv("lloom_csv/media_moksh.csv", index=False)

        print("CSV file 'astro_emotions/media_moksh.csv' has been created successfully!")
    alluniqueframes_ls=list(set(allframes_bigls))
    # Counter method: frequency filter frames
    # all_frames_counter = Counter(allframes_bigls)
    # print('-------Countr------------')
    # print(all_frames_counter)
    # for k,v in all_frames_counter.items():
    #     if v<=15:
    #         alluniqueframes_ls.append(k)
    print('final num of unique ls:', len(alluniqueframes_ls))
    alluniqueframes_array = np.array(alluniqueframes_ls)
    area_ls=[area]*len(alluniqueframes_ls)
    data_dict={'frame': alluniqueframes_ls, 'area':area_ls}
    dataset = Dataset.from_dict(data_dict)
    return alluniqueframes_ls, alluniqueframes_array

def get_data(data_path,info, csv):
    if info:
        print('---------Data Info-------------')
        print("Data path:", data_path)
        print('----------------------')
    


    # Initialize an empty list to store all labels
    allframes_bigls = []

    # Load the JSON data
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Iterate over each entry in the JSON file
    for key, value in data.items():
        if "LLM_Annotation" in value:
            for annotation in value["LLM_Annotation"]:
                if "label" in annotation:
                    # Add the labels to the list
                    allframes_bigls.extend(annotation["label"])
    if csv:
        doc_id=range(1,len(allframes_bigls)+1)
        df = pd.DataFrame({'doc_id': doc_id, 'frames': allframes_bigls})
        df.to_csv("lloom_csv/astro_gen.csv", index=False)
    alluniqueframes_ls=list(set(allframes_bigls))
    alluniqueframes_array = np.array(alluniqueframes_ls)
    return alluniqueframes_ls, alluniqueframes_array

def encode_data(model, data_type, data_path, info):
    ## data prep
    if data_type=='media':
        alluniqueframes_ls, alluniqueframes_array = get_media_frame_data(data_path=data_path, info=info, csv=False)
    elif data_type=='emo' or data_type=='astro':
        alluniqueframes_ls, alluniqueframes_array = get_data(data_path=data_path, info=info, csv=True)
    
    ## get embedding
    embeddings = model.encode(alluniqueframes_ls)
    return embeddings, alluniqueframes_array
    
def cluster_and_getframesegs(embeddings, alluniqueframes_array, k,info=False):
    '''
    k: the number of clusters for kmeans
    '''
    ## Plan: get clustering of all frames to create next stage codebook
    # Steps: data prep -> get embedding -> use embedding to run clustering algo -> get codebook
    # First step->Last step: all frames we currently have -> merged codebook(merged frames)
    # clean data! now i simply exclude them from the dataset 
    
    ## all parameters
    
    ## data prep
    # if data_type=='media':
    #     alluniqueframes_ls, alluniqueframes_array = get_media_frame_data(data_path=data_path, info=info)
    # elif data_type=='emo' or data_type=='astro':
    #     alluniqueframes_ls, alluniqueframes_array = get_data(data_path=data_path, info=info)
    
    # ## get embedding
    # embeddings = model.encode(alluniqueframes_ls)
    
    ## cluster
    kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    if info:
        print('k:',k)
        print('iter #:', kmeans.n_iter_)
    frame_dict={}
    num=0
    for cluster_idx in range(k):
        
        # Get all vectors in the current cluster
        cluster_vectors = embeddings[labels == cluster_idx]
        # Get the frames in this cluster
        cluster_frames = alluniqueframes_array[labels == cluster_idx]
        # Get the centroid for the current cluster
        centroid = centroids[cluster_idx]

        # Calculate the Euclidean distance between each vector in the cluster and the centroid
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        # Find the index of the closest vector
        closest_idx = np.argmin(distances)
        
        # Get the closest frame
        closest_frame = cluster_frames[closest_idx]
        if info:
            print('------------------------------')
            print(">>>>>>>>", closest_frame,"<<<<<<<<")
            print(cluster_frames[:10])
        frame_dict[closest_frame]=list(cluster_frames)
        if len(list(cluster_frames))>num: num = len(list(cluster_frames))
    maxlen_val=num
    return frame_dict, kmeans, maxlen_val
def get_emo_gold_labels(data_path):
    try:
        # Try reading with utf-8 encoding
        data = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If utf-8 fails, try a different encoding
        data = pd.read_csv(data_path, encoding='latin1')  # Use 'latin1' as fallback encoding

    # Initialize an empty dictionary to store motivations
    motivations_dict = {}

    # Iterate over the 'Motivations (select)' column
    for motivations in data['Motivations (select)']:
        if pd.notnull(motivations):
            # Split the motivations by ',' and add to the dictionary
            for motivation in motivations.split(','):
                motivation = motivation.strip()  # Remove extra whitespace
                motivations_dict[motivation] = 'place_holder'

    # Print or use the dictionary
    return motivations_dict
    
def get_astro_gold_labels(data_path):


    # Read the CSV file into a DataFrame
    #df = pd.read_csv(data_path)

    try:
        # Try reading with utf-8 encoding
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If utf-8 fails, try a different encoding
        df = pd.read_csv(data_path, encoding='latin1')  # Use 'latin1' as fallback encoding

    
    # Create a dictionary with 'Open Coding' as keys and 'place_holder' as values
    open_coding_dict = {row['Open Coding']: 'place_holder' for _, row in df.iterrows() if pd.notna(row['Open Coding'])}

    return open_coding_dict
    
def to_file(frame_dict, kmeans, maxlen_val, csv_path, model_path):
    
    ## Pad
    for k,v in frame_dict.items():
        p_num = maxlen_val - len(v)
        p_ls=['']*p_num
        v=v+p_ls
        frame_dict[k]=v
    df = pd.DataFrame(frame_dict)
    df.to_csv(csv_path, index=False)
    with open(model_path, 'wb') as file:
        pickle.dump(kmeans, file)
    print("csv dumped", csv_path)
    print("model dumped", model_path)




def generate_random_string(length=20):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    

if __name__ == "__main__":
    
    ## All configs
    #data_path="/data/afield6/afield6/merging_input_media_Llama3.18b_selSeg/"
    #data_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/"
    #data_path='/data/afield6/afield6/merging_input/astrobot/Llama-3.2-8B-Instruct/e429708c-b1c5-11ef-9d68-7cc25542b43a.json'
    data_path='iter_coder'#'/data/afield6/afield6/merging_input/astrobot/Llama-3.2-8B-Instruct/e429708c-b1c5-11ef-9d68-7cc25542b43a.json'
    data_type='media'
    inductive_coding_method='iter_coder' # 'lloom', 'gen_and_ml_cluster', 'iter_coder'
    print('---------Config------------')
    print('Data type:', data_type)
    print('Data path:', data_path)
    print('Inductive Coding Method:', inductive_coding_method)
    ## get output file paths
    # '/home/pwang71/pwang71/field/corpora_analysis/output1.csv'
    # '/home/pwang71/pwang71/field/corpora_analysis/output1_model.pkl'
    now = datetime.now()
    numeric_datetime = now.strftime("%Y%m%d%H%M%S")
    csv_path = os.getcwd()+"/out/csv/" + numeric_datetime #+ ".csv"
    model_path = os.getcwd()+"/out/model/" + numeric_datetime #+ ".pkl"
    
    ## load embedding model
    embedding_model='all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model).to('cuda')
    
    ## gold_frame_dict={frame: segs in list}
    if data_type=='media': 
        gold_frame_dict=get_gold_labels()
        all_codebook, data_dict=get_iter_codebook(path='/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/slurm-21431843_alldata_timeout.out', data_set=gold_frame_dict)
    
    elif data_type=='astro': 
        gold_frame_dict=get_astro_gold_labels(data_path='astro_emo_gold_csv/astronomy-bot.csv')
        # #gold_frames= ['Bibliometric search with a general topic or theme in mind. This label involves searching for papers or authors within a broad topic.', 
        #                 'A topic that remains unresolved or has multiple conflicting perspectives. This label indicates the complexity or controversy surrounding a particular issue.', 
        #                 'Bot capabilities refer to the analysis and evaluation of automated tools or systems. This label involves assessing the strengths and limitations of bots in a particular context.', 
        #                 'This label describes the process of seeking general knowledge or common sense information. It involves a broad description of a topic or concept.', 
        #                 'Stress testing is a process that involves evaluating the reliability and robustness of a system or tool. This label is often used in the context of bot capabilities or automated systems.', 
        #                 'Deep knowledge refers to in-depth information or analysis that may include opinions or speculations. This label is often used when evaluating complex or nuanced topics.', 
        #                 'This label is used to describe knowledge seeking with a focus on specific factual information. It involves a clear and concise search for concrete data or evidence.', 
        #                 'Bibliometric search with a specific paper or author in mind. This label involves searching for and analyzing a particular paper or author within a broader context.', 
        #                 'CONTESTED refers to a topic that has multiple conflicting perspectives or opinions. This label indicates that a topic is highly debated or contested among experts.', 
        #                 'This label describes the process of seeking knowledge with a focus on specific procedures or methods. It involves a step-by-step analysis of how something is done.']
    elif data_type=='emo': 
        gold_frame_dict=get_emo_gold_labels(data_path='astro_emo_gold_csv/emotions.csv')
        # #gold_frames = ['Unspecified',
        #                 'Affective computing/HCI: A subfield that focuses on the interaction between humans and technology, with a specific emphasis on emotions and affect. This includes areas such as facial expression recognition, sentiment analysis, and human-computer interaction.',
        #                 'Prior work: Refers to existing research or studies that are relevant to the current project or topic, often cited to provide context or background information.',
        #                 'HCI: Chatbots: A subfield of human-computer interaction (HCI) that focuses on designing and developing chatbots, which are computer programs that use natural language processing to simulate conversations with humans.',
        #                 "Other / misc: A catch-all category for any topics that don't fit into the other categories, often including anything that doesn't belong in the other categories.",
        #                 'Healthcare: Mental health: A subfield of healthcare that focuses on mental health, including conditions such as anxiety, depression, and trauma. This category may include research on treatment options, support systems, and preventive measures.',
        #                 'Security: Refers to the protection of computer systems, networks, and data from unauthorized access, use, disclosure, disruption, modification, or destruction.',
        #                 "Healthcare: Neurological disorders: A subfield of healthcare that focuses on conditions that affect the nervous system, including stroke, Parkinson's disease, multiple sclerosis, and epilepsy. This category may include research on diagnosis, treatment, and management of these conditions.",
        #                 'Call center / call screening: A type of interaction between humans and technology that occurs in call centers, where human agents interact with customers using a computer system to handle customer inquiries and resolve issues.']
    gold_frames = list(gold_frame_dict.keys())
    gold_frames_num = len(gold_frames)
    print('>>>>>>>>Gold Frames<<<<<<<<<<<<')
    print(gold_frames)
    print('Number of gold frames:', gold_frames_num)
    
    ## Encode data
    if inductive_coding_method=='gen_and_ml_cluster':
        embeddings, alluniqueframes_array = encode_data(model=model, data_type=data_type, data_path=data_path, info=True)
    
    ## Metrics
    metrics = Metrics(embedding_model=model)
    true_lab_num=len(gold_frames)
    ## Wrong
    print('\nPrecision=thresh>0.5/(num of hat frames - duplicated true positives); Recall=thresh>0.5/number of gold labels\n')
    round=1
    if inductive_coding_method=='gen_and_ml_cluster':
        for k in range(gold_frames_num,2*gold_frames_num,2):
            ## frame_dict={center frame: frames in cluster}
            hat_frame_dict, kmeans, maxlen_val =cluster_and_getframesegs(embeddings=embeddings, alluniqueframes_array=alluniqueframes_array,k=k, info=False)
            ## Save clustering results
            to_file(frame_dict=hat_frame_dict, kmeans=kmeans, maxlen_val=maxlen_val, csv_path=csv_path+'_'+str(round)+'.csv', model_path=model_path+'_'+str(round)+'.pkl')
        
            
            hat_frames = list(hat_frame_dict.keys()) 
            metrics.calculate_metrics(gold_frames=gold_frames, hat_frames=hat_frames)
        
            
            ## WRONG: counting the ones that are predicting the same one multiple times. please fix
            print('---------k:'+str(k)+'-------------')
            
            print('>>>>>Precision:', metrics.get_precision())
            print('>>>>>Recall:', metrics.get_recall())
            metrics.print_metrics_info()
            round+=1
    elif inductive_coding_method=='lloom' or inductive_coding_method=='iter_coder':
        if inductive_coding_method=='lloom':
            # hat_frames=[
            #             "Fact Acquisition",
            #             "Bibliometric Search",
            #             "System Evaluation",
            #             "Bot Overview",
            #             "Broad Knowledge",
            #             "Disagreement Topic",
            #             "Unresolved Topic",
            #             "Knowledge Procedure",
            #             "Factual Inquiry",
            #             "In-depth Expertise",
            #             "Knowledge Perspective"
            #         ]
            ## Lloom moksh frames
            # hat_frames = [ 
            #     "Well-being and Happiness",
            #     "Cultural Heritage",
            #     "Equality and Rights",
            #     "Ethical Considerations",
            #     "Public Opinion Influence",
            #     "Responsibility and Accountability",
            #     "Historical Context and Analysis",
            #     "Immigration Issues",
            #     "Legal Frameworks",
            #     "Health Initiatives",
            #     "Political Dynamics",
            #     "Legal Matters"
            # ]
            ## gpt4oCluster-llama8b-Mian
            # hat_frames = [
            #     "political issues",
            #     "social issues",
            #     "economic issues",
            #     "health issues",
            #     "legal issues",
            #     "crime and justice",
            #     "community issues",
            #     "political dynamics",
            #     "public opinion",
            #     "cultural issues",
            #     "crisis and conflict",
            #     "miscellaneous",
            #     "media and communication",
            #     "activism and advocacy",
            #     "leadership and governance",
            #     "demographics and trends",
            #     "environmental concerns",
            #     "personal issues"
            # ]
            ## Lloom on media article segments
            # hat_frames = [
            #     "public health concerns",
            #     "political legislation",
            #     "community events",
            #     "tobacco regulation",
            #     "health impact of smoking",
            #     "illegal immigration",
            #     "detention and deportation",
            #     "immigrant contributions",
            #     "asylum and refugees",
            #     "immigration enforcement",
            #     "civil unions and rights",
            #     "opposition to gay marriage",
            #     "same-sex marriage issues",
            #     "immigration legal issues",
            #     "tobacco industry economics",
            #     "smoking prevention efforts",
            #     "societal views on gay marriage"
            # ]
            ## lloom on astro gen data
            # hat_frames = [
            #     "Comparison Query",
            #     "Literature Search",
            #     "Irrelevant Content",
            #     "Summary or Overview Request",
            #     "List Request",
            #     "Data Analysis",
            #     "Research Methods",
            #     "Observational Data",
            #     "Information Inquiry",
            #     "User Information Query",
            #     "Clarification Request",
            #     "Theoretical Inquiry"
            # ]
            ## lloom on astro og data
            hat_frames = [
                "Recent Developments",
                "Comparison of Studies",
                "Summarize Literature",
                "Historical Literature Inquiry",
                "Exoplanet Literature",
                "Galaxy Evolution",
                "Hycean Planets",
                "Globular Clusters",
                "Pluto's Status",
                "Transit Lightcurves",
                "Galaxy Clusters",
                "Reionization Effects",
                "Astronomical Data",
                "Massive Galaxies",
                "ADS Citation Counts",
                "Galaxy Mass-Metallicity",
                "Literature Request",
                "Object and Phenomena Literature",
                "Methodology and Models",
                "Observations and Instruments"
            ]




            metrics.calculate_metrics(gold_frames=gold_frames, hat_frames=hat_frames)
            
                
            ## WRONG: counting the ones that are predicting the same one multiple times. please fix
            
            
            print('>>>>>Precision:', metrics.get_precision())
            print('>>>>>Recall:', metrics.get_recall())
            metrics.print_metrics_info()
        elif inductive_coding_method=='iter_coder':
            all_hat_frames=get_iter_codebook(path='/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/slurm-21431843_alldata_timeout.out')
            for i,hat_frames in enumerate(all_hat_frames):
                print('--------iter',i,'-----------')
                metrics.calculate_metrics(gold_frames=gold_frames, hat_frames=hat_frames)
            
                
                ## WRONG: counting the ones that are predicting the same one multiple times. please fix
                
                
                print('>>>>>Precision:', metrics.get_precision())
                print('>>>>>Recall:', metrics.get_recall())
                metrics.print_metrics_info()
            iter_coder_lloom_csv=True # to generate iter_coder's lloom csv
            if iter_coder_lloom_csv:
                for i,hat_frames in enumerate(all_hat_frames):
                    doc_id=range(1,len(hat_frames)+1)
                    df = pd.DataFrame({'doc_id': doc_id, 'frames': hat_frames})
                    df.to_csv("lloom_csv/media_iter_gen"+str(i+1)+'.csv', index=False)
    else:
        raise ValueError('inductive_coding_method is not supported now')
        

    
    ## Gold frames
    #gold_frames = list(gold_frame_dict.keys())
    # hat_frames = list(hat_frame_dict.keys()) 
        
    # # 1. Dictionary with all keys replaced with random strings
    # all_random_keys_dict = {generate_random_string(): value for key, value in hat_frame_dict.items()}

    # # 2. Dictionary with only half keys replaced with random strings
    # keys = list(hat_frame_dict.keys())
    # half_length = len(keys) // 2
    # random_keys = keys[:half_length]  # First half to replace
    # unchanged_keys = keys[half_length:]  # Second half to keep the same

    # half_random_keys_dict = {
    #     (generate_random_string() if key in random_keys else key): value
    #     for key, value in hat_frame_dict.items()
    # }
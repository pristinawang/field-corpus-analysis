import re
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

def get_iter_codebook(path, data_set=None):
    texts=[]
    labels=[]
    l=0
    if data_set is not None:
        for k, vs in data_set.items():
            for v in vs:
                labels.append(l)
                texts.append(v)
            l+=1
        unique_values = set(labels)

        # Count unique values
        
        if len(unique_values) < len(texts):
            metrics.calculate_cluster_metrics(texts=texts, labels=labels)
            
            print(metrics.silhouette_score)
        return
    all_codebook=[]
    with open(path, "r") as file:
        content = file.read()  # Reads the whole file as a string
    content_ls=content.split('Iter:')[1:]
    data_dict={}
    embedding_model='all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model).to('cuda')
    metrics = Metrics(embedding_model=model)
    for i,iter in enumerate(content_ls):
        iter_ls=iter.split('---------')
        code_book=iter_ls[1].split('----')[0]
        #matches = re.findall(r'\d+\.\s(.*?)\n\n->', code_book)
        
        matches = re.findall(r"(\d+)\.\s+(.+?)\n->\s+(.+?)(?=\n\d+\.|\Z)", code_book, re.DOTALL)
        for match in matches:
            key = int(match[0])  # Extracting the integer key
            label = match[1].strip()  # Extracting the label
            content = match[2].strip()  # Extracting the text after '->'
            data_dict[key-1] = content.split('->')
        # Extract only the parts after the number
        #extracted_texts = [match[1].strip() for match in matches]
        # if i==1:
        #     #print(matches)
        #     print(code_book)
        #     print(data_dict)
        
        
        ## Si score example
        # texts=[]
        # labels=[]
        # for k,vs in data_dict.items():
        #     for v in vs:
        #         labels.append(k)
        #         texts.append(v)
        # unique_values = set(labels)

        # # Count unique values
        # print('-------Si'+str(i)+'----------')
        # if len(unique_values) < len(texts):
        #     metrics.calculate_cluster_metrics(texts=texts, labels=labels)
            
        #     print(metrics.silhouette_score)
        # all_codebook.append(matches)
    return all_codebook, data_dict
        
        

if __name__ == "__main__":
    all_codebook, data_dict=get_iter_codebook(path='/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/slurm-21431843_alldata_timeout.out')
    

 
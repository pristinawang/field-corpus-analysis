import json
import pandas as pd
import ast

def get_media_gold_labels():
    '''
    input_segments: return: dict type; key: article id, value: article text
    '''
    codes=getcodes()
    for k,v in codes.items():
        key=k
        k=float(k)
        primek=str(round(k))+'.0'
        if k>=1.0 and k<=15.2:
            codes[key]=codes[primek]
    newcodes={}
    for k,v in codes.items():
        newcodes[float(k)]=v
    codes=newcodes

    frame_dict={}
    frame_arid_dict={}
    article_dict={}
    for k,v in codes.items():
        frame_dict[v]=[]
        frame_arid_dict[v]=set() #key is gold_code, value is set of article ids
    paths=['/data/afield6/afield6/moksh/media_frames_corpus/tobacco.json'
        ,'/data/afield6/afield6/moksh/media_frames_corpus/immigration.json'
        ,'/data/afield6/afield6/moksh/media_frames_corpus/samesex.json']
    art=set()
    allart=set()
    ## Process files
    # Read the JSON file
    for path in paths:
        with open(path, 'r') as file:
            data = json.load(file)
        for k, article in data.items():
            text=data[k]['text']
            irrelevants=data[k]['annotations']['irrelevant']

            for annotator, segs in data[k]['annotations']['framing'].items():
                if not(irrelevants[annotator]):
                    for seg in segs:
                        frame=codes[seg['code']]
                        segment=text[seg['start']:seg['end']].strip('\n')
                        segment=segment.replace('\n\n', ' ')
                        frame_dict[frame].append(segment)
                        frame_arid_dict[frame].add(k) # key is code; value is set of article ids
                        art.add(k)
                    article_dict[k]=text
            allart.add(k)
    keys_to_remove=[]
    ## Remove PRIMARY
    for key in frame_dict:
        frame_dict[key] = [item for item in frame_dict[key] if item != "PRIMARY"]
        if len(frame_dict[key])==0: 
            keys_to_remove.append(key)
    
    ## Remove length zero k,v pairs
    for key in keys_to_remove:
        if key in frame_dict:
            del frame_dict[key]
            del frame_arid_dict[key]
    print('All art', len(allart), 'rel art', len(art))  
    return frame_dict, frame_arid_dict, article_dict # key is code; value is set of article ids

def get_data_and_labels(data_type):
    if data_type=='media':
        return get_media_gold_labels()
    elif data_type=='astro':
        return get_astro_gold_labels()
    elif data_type=='emo':
        return get_emo_gold_labels()
    elif data_type=='val_e':
        return get_expli_values_gold_labels()
    elif data_type=='acl':
        return get_acl_data()
    else: raise ValueError("Data type not supported yet")
    
def get_astro_gold_labels(csv_path='/home/pwang71/pwang71/field/corpora_analysis/astro_emo_gold_csv/astronomy-bot.csv'):
    '''
    return:
    label_dict: gold_label as key and values as the segments,txt, labeled with this gold_label
    label_id_dict: gold_label as key; value is a set of ids, thread_ts for astronomy-bot
    data_dict: use as dataset; key is id(astro-bot id is thread_ts), value is the associated text(full user query for astro-bot)
    '''
    # Load CSV file
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df[::-1].dropna(how='all')[::-1].reset_index(drop=True)
    df['thread_ts'] = df['thread_ts'].astype(int).astype(str)
    
    # Get the values under "Open Coding" column as a list
    labels = df["Open Coding"].dropna().tolist()
    ids = df["thread_ts"].dropna().tolist()
    
    # Construct empty dicts
    label_dict = {label: [] for label in list(set(labels))}
    label_id_dict = {label: set() for label in list(set(labels))}
    data_dict = {thread_id:'' for thread_id in list(set(ids))}
    
    # Loop through rows
    for index, row in df.iterrows():
        thread_id=row["thread_ts"]
        data=row["full_user_query"]
        label=row["Open Coding"]
        label_dict[label].append(data)
        label_id_dict[label].add(thread_id)
        data_dict[thread_id]=data
    return label_dict, label_id_dict, data_dict

def get_emo_gold_labels(csv_path='/home/pwang71/pwang71/field/corpora_analysis/astro_emo_gold_csv/data_pre-processed.csv'):
    '''
    return:
    label_dict: gold_label as key and values as the segments,txt, labeled with this gold_label
    label_id_dict: gold_label as key; value is a set of ids, thread_ts for astronomy-bot
    data_dict: use as dataset; key is id(astro-bot id is thread_ts), value is the associated text(full user query for astro-bot)
    '''
    # Load CSV file
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df[::-1].dropna(how='all')[::-1].reset_index(drop=True)
    df['paperId'] = df['paperId'].astype(str)
    all_labels=[]
    for row in df["motivations"].dropna():
        labels=ast.literal_eval(row)
        all_labels.extend(labels)

    labels = list(set(all_labels))
    ids = df["paperId"].dropna().tolist()
    
    # Construct empty dicts
    label_dict = {label: [] for label in labels}
    label_id_dict = {label: set() for label in labels}
    data_dict = {thread_id:'' for thread_id in list(set(ids))}
    
    # Loop through rows
    for index, row in df.iterrows():
        paper_id=row["paperId"]
        data=row["motivation_free_txt"]
        labels=ast.literal_eval(row["motivations"])
        for label in labels:
            label_dict[label].append(data)
            label_id_dict[label].add(paper_id)
        data_dict[paper_id]=data
    return label_dict, label_id_dict, data_dict

def get_expli_values_gold_labels(csv_path='/home/pwang71/pwang71/field/corpora_analysis/astro_emo_gold_csv/uplifted_values_explicit.csv'):
    
    # Load CSV file
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df[::-1].dropna(how='all')[::-1].reset_index(drop=True)
    df['quote_id'] = df['quote_id'].astype(int).astype(str)
    all_labels=[]
    for row in df["labels"].dropna():
        labels=ast.literal_eval(row)
        all_labels.extend(labels)
    
    labels = list(set(all_labels))
    ids = df["quote_id"].dropna().tolist()
    
    # Construct empty dicts
    label_dict = {label: [] for label in labels}
    label_id_dict = {label: set() for label in labels}
    data_dict = {q_id:'' for q_id in ids}
    
    # Loop through rows
    for index, row in df.iterrows():
        q_id=row["quote_id"]
        data=row["quotation"]
        labels=ast.literal_eval(row["labels"])
        for label in labels:
            label_dict[label].append(data)
            label_id_dict[label].add(q_id)
        data_dict[q_id]=data
    return label_dict, label_id_dict, data_dict

def get_acl_data(csv_path='/home/pwang71/pwang71/field/corpora_analysis/astro_emo_gold_csv/acl.csv'):
    df = pd.read_csv(csv_path, encoding='latin1')
    df = df[::-1].dropna(how='all')[::-1].reset_index(drop=True)
    data_dict={}
    for index, row in df.iterrows():
        p_id=row["paper_id"]
        data=row["abstract"]
        data_dict[p_id]=data
    return None, None, data_dict

def getcodes(path='/data/afield6/afield6/moksh/media_frames_corpus/codes.json'):
    with open(path, 'r') as file:
        codes = json.load(file)
    return codes
if __name__=='__main__':
    get_emo_gold_labels()
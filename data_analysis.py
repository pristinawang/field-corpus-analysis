import json
import os
import matplotlib.pyplot as plt
from collections import Counter

def get_og_codes(code_path):


    # Load JSON from a file
    with open(code_path, 'r') as file:
        dictionary = json.load(file)

    # Reverse keys and values
    reversed_dict = {value: key for key, value in dictionary.items()}

    # Print the reversed dictionary
    return reversed_dict

def get_gen_codes(directory_path):
    # with open(frame_path, 'r') as file:
    #     data = json.load(file)
    
    # # Extract frames into a new dictionary
    # result = {key: [annotation['frame'] for annotation in annotations['LLM_Annotation']] for key, annotations in data.items()}
    
    # return result

    """Load JSON from all files in a directory and extract frames into a single dictionary."""
    combined_dict = {}
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):  # Ensure it's a JSON file
            file_path = os.path.join(directory_path, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Extract frames and add them to the combined dictionary
                for key, annotations in data.items():
                    frames = [annotation['frame'] for annotation in annotations['LLM_Annotation']]
                    combined_dict[key] = frames
    
    return combined_dict

def calculate_overlap(codes, frames):
    """Calculate the overlap of frames between two dictionaries and return a sorted dictionary."""
    frames1=codes
    frames2=frames
    overlap_dict = {}

    # # Iterate over each key in frames1
    # for key1, frames_list1 in frames1.items():
    #     max_overlap = 0
    #     max_key = None
    frames_list1=list(codes.keys())


    # Compare with every key in frames2
    for key2, frames_list2 in frames2.items():
        if not isinstance(frames_list1, list):
            print(f"Warning: Expected list for key '{key1}', but got {type(frames_list1)}. Skipping...")
            continue
        if not isinstance(frames_list2, list):
                print(f"Warning: Expected list for key '{key2}', but got {type(frames_list2)}. Skipping...")
                continue
        if isinstance(frames_list2, list) and isinstance(frames_list1, list):
        # Calculate the intersection (overlap) of frames
            s1=set(frames_list1)
            try:
                s2=set(frames_list2)
                overlap = len(set(frames_list1) & set(frames_list2))
                overlap_dict[key2] = overlap
            except Exception as e:
                print(key2,frames_list2)
            
            


    # Sort the dictionary by the overlap count in descending order
    sorted_overlap_dict = {k: v for k, v in sorted(overlap_dict.items(), key=lambda item: item[1], reverse=True)}

    return sorted_overlap_dict

# def getallframes(dir_path):
#     '''
#     return list type, all gen frames from all articles in this directory
#     return list, return ids of all articles that have their frames generated
#     ex input: dir_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/immigration"
#     '''
#     allframes=[] #all frames from all articles
#     gen_articleids=[]
#     for filename in os.listdir(dir_path):
#         if filename.endswith(".json"):  # Ensure it's a JSON file
#             file_path = os.path.join(dir_path, filename)
            
#             # Load the JSON file
#             with open(file_path, 'r') as file:
#                 data = json.load(file)
                
#                 # Extract frames and add them to the combined dictionary
#                 for article_id, annotations in data.items():
#                     for annotation in annotations['LLM_Annotation']:
#                         if not(isinstance(annotation['frame'], str)):
#                             print(filename,article_id, annotation['frame'])
#                         else:
#                             allframes.append(annotation['frame'])
#                     gen_articleids.append(article_id)
    
#     return allframes, gen_articleids

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
                        if not(isinstance(annotation['frame'], str)) or annotation['frame']=='':
                            if printerrorframes: print(filename,article_id, annotation['frame'])
                        else:
                            allframes.append(annotation['frame'])
                    gen_articleids.append(article_id)
    
    return allframes, gen_articleids

def getallarticleids(metadata_path):
    '''
    return type: list
    return all original article ids from this subgroup(immigration or samesex or tobacco)
    all article ids from this metadata file(because getting from metadat so these are all article ids, no more no less)
    
    input is metadata directory path
    '''
    article_ids=[]
    with open(metadata_path, 'r') as file:
        data = json.load(file)
                
        # Extract frames and add them to the combined dictionary
        for article_id, _ in data.items():
            article_ids.append(article_id)
    return article_ids

def plotfrequency(strings_list, out_dir="/home/pwang71/pwang71/field/corpora_analysis/pics/"):
    # Clean the list to make sure all "$" get turned into "\$"
    strings_list = [s.replace("$", "\$") for s in strings_list]
    # Count the frequencies
    string_counts = Counter(strings_list)

    # Separate the strings and their counts for plotting
    labels = list(string_counts.keys())
    frequencies = list(string_counts.values())

    # Plotting
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    plt.bar(labels, frequencies, color='skyblue')
    plt.xlabel('Strings')
    plt.ylabel('Frequency')
    plt.title('Frequency of Distinct Strings')
    plt.savefig(out_dir+"freq_gen_frames.png")
    
def frequency(my_list):
    count=0
    for frame in my_list:
        frame=frame.lower()
        if "quality" in frame and "life" in frame:
            count+=1
    print('quality of life #:',count)
    print(Counter(my_list))



if __name__ == "__main__":
    # code_path="/data/afield6/afield6/moksh/media_frames_corpus/codes.json"
    # code_dict=get_og_codes(code_path=code_path)
    # frame_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/immigration/immigration1.json"
    # dir_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/immigration"
    # frames=get_gen_codes(dir_path)
    # overlap_dict=calculate_overlap(codes=code_dict, frames=frames)
    # print(overlap_dict)

    ## Loop through generated code from one folder
    
    areas=['immigration', 'tobacco', 'samesex']
    for area in areas:
        dir_path="/data/afield6/afield6/moksh/mediaframes_output_data/seg_frames/"+area
        gen_frames_immigration, gen_articleids=getallframes(dir_path=dir_path)
        metadata_article_ids=getallarticleids("/data/afield6/afield6/moksh/media_frames_corpus/"+area+"_metadata.json")
        article_ids=getallarticleids(metadata_path="/data/afield6/afield6/moksh/media_frames_corpus/"+area+".json")
        print('#################################')
        print('Area:', area)
        print('-------number of articles--------')
        print(area+'_metadata.json #:', len(metadata_article_ids), len(set(metadata_article_ids)))
        print(area+'.json #:', len(article_ids), len(set(article_ids)))
        print('gen #:', len(gen_articleids), len(set(gen_articleids)))
        print('**Notes: '+area+'_metadata.json'+' counted using '+area+'_metadata.json'+', another '+area+'.json'+' used '+area+'.json')        
        print('-------number of gen frames--------')
        print("total gen #:", len(gen_frames_immigration))
        print("unique #:", len(set(gen_frames_immigration)))

    
    
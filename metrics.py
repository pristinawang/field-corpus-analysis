import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, os
import math

def safe_equal(a, b):
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b

class Metrics:
    '''
    call calculate_metrics first and get_precision, get_recall
    '''
    def __init__(self, embedding_model):
        self.embedding_model=embedding_model
        self.precision=None
        self.recall=None
        self.f1=None
        self.normalized_mutual_information=None
        self.silhouette_score=None
        self.sim_dict=None
        self.sim_matrix=None
    
    def get_precision(self):
        if self.precision is None:
            print("Metrics haven't been set yet. Use calculate_metrics method to set metrics and then get_precision after that.")
        else:
            return self.precision
    
    def get_recall(self):
        if self.recall is None:
            print("Metrics haven't been set yet. Use calculate_metrics method to set metrics and then get_recall after that.")
        else:
            return self.recall
    
    def print_metrics_info(self):
        if self.recall is None or self.precision is None:
            print("Metrics haven't been calculated yet. Use calculate_metrics method to set metrics first.")
        else:
            ## True positive
            print('------------------------')
            print("True positive(hat frames with cos sim scores between most similar gold frames greater than thresh)")
            for gold_frame, hat_frames in self.true_positives_dict.items():
                print('Gold Frame:', gold_frame, ',Hat frames:', '; '.join(hat_frames))
            ## False negative
            print('------------------------')
            print("False negative:(gold frames that are not predicted)")
            for gold_frame in list(self.false_negatives):
                print('Gold frame:', gold_frame)
            ## False positive
            print('------------------------')
            print("False positive(hat frames with cos sim scores between most similar gold frames lesser than thresh)")
            for hat_frame in self.false_positives:
                print('Hat frame:',hat_frame, ',Most similar gold frame of this hat frame:', self.sim_dict[hat_frame][0])
            print('------------------------')
    def calculate_cluster_metrics(self, frame_seg_dict):
        '''
        input: any dictionary that has frame(str) as key, list of segments(str) as value
        
        ---------
        Convert into texts and labels then calculate
        
        texts: is a list of strings(could be og text or labels)
        labels: integer list representing the cluster that each text is assigned to
        len(texts)==len(labels)
        '''
        texts=[] # [str]
        labels=[] # [int]
        for i, (frame, segs) in enumerate(frame_seg_dict.items()):
            for seg in segs:
                labels.append(i)
                texts.append(seg)
                
        assert len(texts) == len(labels), "Error: texts and labels must have the same length"
        texts_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, device='cuda')  # Shape: (len(hat_frames), embedding_dim)
        assert len(texts_embeddings) == len(labels), "Error: texts_embeddings and labels must have the same length"
        assert len(set(labels)) < len(texts), "Error: number of unique labels must be smaller than number of text in texts"

        return silhouette_score(texts_embeddings.cpu().numpy(), labels)
    
    def compute_f1(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def save_table_as_png(self,df, title, output_path):
        if df.empty:
            print("⚠️ Warning: DataFrame",title," is empty. Skipping table rendering.")
            return
        fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(df))))  # Dynamic height
        ax.axis('off')
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        rowLabels=df.index,
                        loc='center',
                        cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust row height
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    def calculate_pairwise_overlap(self, gold_frame_article_id_dict, hat_frame_article_id_dict_ls, cos_sim_thresh=0.5):
        hat_frames_ls=[]
        for hat_frame_article_id_dict in hat_frame_article_id_dict_ls:
            hat_frames_ls.append(list(hat_frame_article_id_dict.keys()))
        gold_frames = list(gold_frame_article_id_dict.keys())
        # New calculation
        sim_matrices_thresh=[]
        for hat_frames in hat_frames_ls:
            sim_matrix=self.get_sim_frame(gold_frames=gold_frames, hat_frames=hat_frames, reverse_matrix=True) #reverse_matrix: rows are gold frames
            sim_matrices_thresh.append(torch.where(sim_matrix >= cos_sim_thresh, sim_matrix, float('nan')))
        
        # Add overlap
        hat_include_ls=[[] for _ in hat_frames_ls]
        gold_include_ls=[[] for _ in hat_frames_ls]
        for k,hat_frames in enumerate(hat_frames_ls):
            # Precision per hat_frame
            for j, hat_frame in enumerate(hat_frames):
                col_vec = sim_matrices_thresh[k][:, j]   # all rows (gold), single column

                # if not(torch.isnan(col_vec).all()):
                #     hat_include_ls[k].append(hat_frame)
                    
                if not(torch.isnan(col_vec).all()):
                    # Get indices where the value is NOT NaN (i.e., passed threshold)
                    valid_indices = torch.isnan(col_vec) == False
     
                    
                    # Multi-mapping: the set of article ids for all gold frames that this hat_frame maps to

                    for i in torch.where(valid_indices)[0]:
                        gold_frame = gold_frames[i]
            
                        hat_include_ls[k].append(gold_frame)
                
            
            for i, gold_frame in enumerate(gold_frames):
                row_vec = sim_matrices_thresh[k][i]

                if not(torch.isnan(row_vec).all()):
                    gold_include_ls[k].append(gold_frame)
        
        print('Path 1 has precision true positives', hat_include_ls[0])
        print('Path 2 has precision true positives', hat_include_ls[1])
        print('Path 1 has recall true positives', gold_include_ls[0])
        print('Path 2 has recall true positives', gold_include_ls[1])
        hat_include_master = set(hat_include_ls[0]) & set(hat_include_ls[1])
        gold_include_master = set(gold_include_ls[0]) & set(gold_include_ls[1])
        return {'hat_include_master': hat_include_master, 'gold_include_master': gold_include_master}
        
    def calculate_article_metrics(self, gold_frame_article_id_dict, hat_frame_article_id_dict, article_ids_seen=None, 
                                    hat_include_master=None, gold_include_master=None, info=False, all_seen=True, cos_sim_thresh=0.5, output_dir_path=None):
        # hat_frame_article_id_dict: hat_frame as k and set of article ids as v
        # gold_frame_article_id_dict: k is gold labels and v is set of article ids
        # sim_dict: key is hat frame : value is tuple of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        if not all_seen:
            assert not(article_ids_seen is None), f"article_ids_seen cannot be None when all_seen is False"
            # Trim gold_frame_article_id_dict to article_ids that have been seen by model
            new_gold_frame_article_id_dict={ gold_frame : set() for gold_frame in gold_frame_article_id_dict.keys()}
            for gold_frame, article_ids in gold_frame_article_id_dict.items():
                for article_id in list(article_ids):
                    if article_id in article_ids_seen:
                        new_gold_frame_article_id_dict[gold_frame].add(article_id)
            # Remove gold_frames with zero article ids, aka remove gold_frames that weren't seen
            new_gold_frame_article_id_dict = {k: v for k, v in new_gold_frame_article_id_dict.items() if len(v) > 0}
            gold_frame_article_id_dict=new_gold_frame_article_id_dict
        
        hat_frames = list(hat_frame_article_id_dict.keys())
        gold_frames = list(gold_frame_article_id_dict.keys())
        if hat_include_master is None: hat_include_master = hat_frames
        else: print(f"⚠️ Pairwise precision is being calculated with only{hat_include_master}")
        if gold_include_master is None: gold_include_master = gold_frames
        else: print(f"⚠️ Pairwise recall is being calculated with only{gold_include_master}")
        
        # New calculation
        self.sim_matrix=self.get_sim_frame(gold_frames=gold_frames, hat_frames=hat_frames, reverse_matrix=True) #reverse_matrix: rows are gold frames
        sim_matrix_thresh = torch.where(self.sim_matrix >= cos_sim_thresh, self.sim_matrix, float('nan'))
        
        precisions=[]
        precision_w=[]
        recalls=[]
        recall_w=[]
        metrics_dict={}
        precision_dict={}
        recall_dict={}
        # Precision per hat_frame
        for j, hat_frame in enumerate(hat_frames):
            col_vec = sim_matrix_thresh[:, j]   # all rows (gold), single column

            if not(torch.isnan(col_vec).all()) and hat_frame in hat_include_master:
                # Get indices where the value is NOT NaN (i.e., passed threshold)
                valid_indices = torch.isnan(col_vec) == False
                # All article_ids labeled with this hat_frame
                hat_art_ids=set(hat_frame_article_id_dict[hat_frame])
                
                # Multi-mapping: the set of article ids for all gold frames that this hat_frame maps to
                gold_art_ids=set()
                for i in torch.where(valid_indices)[0]:
                    gold_frame = gold_frames[i]
                    gold_art_ids.update(gold_frame_article_id_dict[gold_frame])
                
                hat_dict={}
                hat_dict['true_positive'] = len(hat_art_ids & gold_art_ids)
                hat_dict['precision'] = hat_dict['true_positive'] / len(hat_art_ids)
                precision_dict[hat_frame] = hat_dict
                precisions.append(hat_dict['precision'])
                precision_w.append(len(hat_art_ids))
                if info:
                    print('---Precision info---------')
                    print('Hat frame:', hat_frame)
                    print('True positives:', hat_art_ids & gold_art_ids)
        
        # Recall per gold_frame
        for i, gold_frame in enumerate(gold_frames):
            row_vec = sim_matrix_thresh[i]

            if not(torch.isnan(row_vec).all()) and gold_frame in gold_include_master:
                # All article_ids labeled with this gold_frame
                gold_art_ids=set(gold_frame_article_id_dict[gold_frame])
                
                # Multi-mapping: the set of article ids for all hat frames that this gold_frame maps to
                hat_art_ids=set()
                valid_indices = torch.isnan(row_vec) == False
                for i in torch.where(valid_indices)[0]:
                    hat_frame = hat_frames[i]
                    hat_art_ids.update(hat_frame_article_id_dict[hat_frame])
                gold_dict={}
                gold_dict['true_positive'] = len(hat_art_ids & gold_art_ids)
                gold_dict['recall'] = gold_dict['true_positive'] / len(gold_art_ids)
                recall_dict[gold_frame] = gold_dict
                recalls.append(gold_dict['recall'])
                recall_w.append(len(gold_art_ids))
                if info:
                    print('---Recall info---------')
                    print('Gold frame:', gold_frame)
                    print('True positives:', hat_art_ids & gold_art_ids)
        if len(precisions)>0: avg_precision=np.average(precisions, weights=precision_w) #sum(precisions)/len(precisions)
        else: avg_precision=float('nan')
        if len(recalls)>0: avg_recall=np.average(recalls, weights=recall_w) #sum(recalls)/len(recalls)
        else: avg_recall=float('nan')
        metrics_dict['avg']={'precision': avg_precision, 'recall': avg_recall}    
        self.article_metric_dict = metrics_dict
        
        if output_dir_path is not None:
            os.makedirs(output_dir_path, exist_ok=True)
            precision_path=os.path.join(output_dir_path, f'seg_precision_table.png')
            recall_path=os.path.join(output_dir_path, f'seg_recall_table.png')
            precision_df = pd.DataFrame.from_dict(precision_dict, orient='index')
            recall_df = pd.DataFrame.from_dict(recall_dict, orient='index')

            # 2. Optional: sort or format
            # precision_df = precision_df.sort_values(by='precision', ascending=False)
            # recall_df = recall_df.sort_values(by='recall', ascending=False)
            try:
                self.save_table_as_png(precision_df, "Segment Level Precision per Hat Frame", precision_path)
            except Exception as e:
                print('⚠️ segment level precision table cannot be generated.', e)
            try:
                self.save_table_as_png(recall_df, "Segment Level Recall per Gold Frame", recall_path)
            except Exception as e:
                print('⚠️ segment level recall table cannot be generated.', e)
        
    def get_sim_frame(self, gold_frames, hat_frames, reverse_matrix=False):
        '''
        input: 
        gold_frames: data type is list, list of gold frames
        hat_frames: data type is list, list of predicted frames
        output:
        sim_dict: keys are all hat frames, values are pairs of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        
        for each hat_frame in hat_frames: find the most similar gold_frame and the cos sim score
        
        if reverse_matrix==true: return cos_sim_matrix and return it using gold_frames as rows of the matrix
        '''

        # Get embeddings for both lists
        hat_frames_embeddings = self.embedding_model.encode(hat_frames, convert_to_tensor=True, device='cuda')  # Shape: (len(hat_frames), embedding_dim)
        gold_frames_embeddings = self.embedding_model.encode(gold_frames, convert_to_tensor=True, device='cuda')  # Shape: (len(gold_frames), embedding_dim)

        # Normalize embeddings for cosine similarity
        hat_frames_embeddings = F.normalize(hat_frames_embeddings, p=2, dim=1)
        gold_frames_embeddings = F.normalize(gold_frames_embeddings, p=2, dim=1)

        if reverse_matrix:
            # Compute cosine similarity between each gold_frame and all hat_frames
            cos_sim_matrix = torch.matmul(gold_frames_embeddings, hat_frames_embeddings.T)  # Shape: (len(gold_frames), len(hat_frames))

            return cos_sim_matrix
        else:
            # Compute cosine similarity between each hat_frame and all gold_frames
            cos_sim_matrix = torch.matmul(hat_frames_embeddings, gold_frames_embeddings.T)  # Shape: (len(hat_frames), len(gold_frames))

            # Find the most similar gold_frame for each hat_frame
            sim_dict = {}
            for i, hat_frame in enumerate(hat_frames):
                best_match_idx = torch.argmax(cos_sim_matrix[i]).item()
                most_similar_gold_frame = gold_frames[best_match_idx]
                cos_sim_score = cos_sim_matrix[i, best_match_idx].item()
                sim_dict[hat_frame] = (most_similar_gold_frame, cos_sim_score)
                
            return sim_dict
    def drop_label_helper_article_dict(self, label_to_drop, gold_frame_article_id_dict, hat_frame_article_id_dict, sim_thresh=0.5):
        
        if label_to_drop is None:
            return {
                    'gold_frame_article_id_dict':gold_frame_article_id_dict,
                    'hat_frame_article_id_dict': hat_frame_article_id_dict
                }
        if not(label_to_drop in gold_frame_article_id_dict):
            raise ValueError("label_to_drop is not in dict")
        # sim_dict: key is hat frame : value is tuple of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        sim_dict=self.get_sim_frame(gold_frames=list(gold_frame_article_id_dict.keys()), hat_frames=list(hat_frame_article_id_dict.keys()))
        
        del gold_frame_article_id_dict[label_to_drop]
        print(f'🚀 gold label "{label_to_drop}" has been dropped from gold dicts')
        
        
        hat_frames_to_dropped=[]
        for hat_frame, tup in sim_dict.items():
            gold_frame = tup[0]
            sim_score = tup[1]
            if gold_frame==label_to_drop and sim_score>sim_thresh: hat_frames_to_dropped.append(hat_frame)
        for hat_label_to_drop in hat_frames_to_dropped:
            del hat_frame_article_id_dict[hat_label_to_drop]
            print(f'🚀 hat label "{hat_label_to_drop}" similar to "{label_to_drop}" has been dropped from hat dicts')
        return {
                    'gold_frame_article_id_dict':gold_frame_article_id_dict,
                    'hat_frame_article_id_dict': hat_frame_article_id_dict
                }  
        
    def drop_label_helper(self, label_to_drop, gold_frame_article_id_dict, hat_frame_article_id_dict, hat_frame_dict, gold_frame_dict, sim_thresh=0.5):
        
        if label_to_drop is None:
            return {
                    'gold_frame_dict':gold_frame_dict,
                    'gold_frame_article_id_dict':gold_frame_article_id_dict,
                    'hat_frame_dict':hat_frame_dict,
                    'hat_frame_article_id_dict': hat_frame_article_id_dict
                }
        if not(label_to_drop in gold_frame_article_id_dict) or not(label_to_drop in gold_frame_dict):
            raise ValueError("label_to_drop is not in dict")
        # sim_dict: key is hat frame : value is tuple of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        sim_dict=self.get_sim_frame(gold_frames=list(gold_frame_dict.keys()), hat_frames=list(hat_frame_dict.keys()))
        
        del gold_frame_article_id_dict[label_to_drop]
        del gold_frame_dict[label_to_drop]
        print(f'🚀 gold label "{label_to_drop}" has been dropped from gold dicts')
        
        
        hat_frames_to_dropped=[]
        for hat_frame, tup in sim_dict.items():
            gold_frame = tup[0]
            sim_score = tup[1]
            if gold_frame==label_to_drop and sim_score>sim_thresh: hat_frames_to_dropped.append(hat_frame)
        for hat_label_to_drop in hat_frames_to_dropped:
            del hat_frame_article_id_dict[hat_label_to_drop]
            del hat_frame_dict[hat_label_to_drop]
            print(f'🚀 hat label "{hat_label_to_drop}" similar to "{label_to_drop}" has been dropped from hat dicts')
        return {
                    'gold_frame_dict':gold_frame_dict,
                    'gold_frame_article_id_dict':gold_frame_article_id_dict,
                    'hat_frame_dict':hat_frame_dict,
                    'hat_frame_article_id_dict': hat_frame_article_id_dict
                }  

    def calculate_precision_recall(self,gold_frames, hat_frames, frame_level_qualitative_result_path, info=False, cos_sim_thresh=0.5):
        '''
        embedding_model: the model use to encode frames/labels to vectors
        gold_frames: data type is list, list of gold frames
        hat_frames: data type is list, list of predicted frames
        
        precision and recall both allow duplications
        '''
        # sim_dict: key is hat frame : value is tuple of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        self.sim_matrix=self.get_sim_frame(gold_frames=gold_frames, hat_frames=hat_frames, reverse_matrix=True) #reverse_matrix: rows are gold frames
        sim_matrix_thresh = torch.where(self.sim_matrix >= cos_sim_thresh, self.sim_matrix, float('nan'))

        true_positives_allow_dup_precision=[]
        #true_positives_set_precision = set()
        for j, hat_frame in enumerate(hat_frames):
            col_vec = sim_matrix_thresh[:, j]   # all rows (gold), single column

            if not(torch.isnan(col_vec).all()):
                true_positives_allow_dup_precision.append(hat_frame)
                
                # Get indices where the value is NOT NaN (i.e., passed threshold)
                # valid_indices = torch.isnan(col_vec) == False
                # matching_gold = [gold_frames[i] for i in torch.where(valid_indices)[0]]
                # Add to global set
                #true_positives_set_precision.update(matching_gold)
        true_positives_multimap_recall=[]
        #true_positives_unique_recall = set()
        for i, gold_frame in enumerate(gold_frames):
            row_vec = sim_matrix_thresh[i]
            # Check if the entire column is NaN
            # if torch.isnan(row_vec).all():
            #     print(f"All values are below threshold (NaN)")
            # else:
            if not(torch.isnan(row_vec).all()):
                true_positives_multimap_recall.append(gold_frame)
                
                # valid_indices = torch.isnan(row_vec) == False
                # matching_hat = [hat_frames[i] for i in torch.where(valid_indices)[0]]
                # true_positives_unique_recall.update(matching_hat)
        
        self.precision = len(true_positives_allow_dup_precision) / len(hat_frames)
        #self.precision_unique = len(true_positives_set_precision) / len(hat_frames)
        self.recall = len(true_positives_multimap_recall) / len(gold_frames)
        #self.recall_unique = len(true_positives_unique_recall) / len(gold_frames)
        if info: 
            print('Frame precision true positives:', true_positives_allow_dup_precision)
            print('Frame recall true positives:', true_positives_multimap_recall)
        if frame_level_qualitative_result_path is not None:
            try:
                output_dir = os.path.dirname(frame_level_qualitative_result_path)
                os.makedirs(output_dir, exist_ok=True)
                # Convert to numpy for plotting
                sim_numpy = sim_matrix_thresh.cpu().numpy()
                sim_numpy = self.sim_matrix.cpu().numpy()

                # Create DataFrame for seaborn heatmap
                df = pd.DataFrame(sim_numpy, index=gold_frames, columns=hat_frames)

                # Plot heatmap
                plt.figure(figsize=(1*len(hat_frames), 0.5*len(gold_frames)))#16,4 #20,20
                ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="flare", cbar=True, linewidth=.5, linecolor="black")
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                plt.title("Gold vs. Hat Frame Cosine Similarity")
                plt.xlabel("Hat Frames (Predicted)")
                plt.ylabel("Gold Frames (True)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(frame_level_qualitative_result_path, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print('⚠️ Heat map cannot be generated.', e)
    def calculate_pairwise_article_metrics(self, gold_frame_article_id_dict, hat_frame_article_id_dict_ls,label_to_drop=None,
                                            segment_level_qualitative_result_output_dir_path=None, info_seg=False):
        gold_frame_article_id_dict_ls=[gold_frame_article_id_dict,gold_frame_article_id_dict]
        
        if not(label_to_drop is None):
            print('⛔ Verify the first and only the first file_path is topicgpt file_path')
            data_dict= self.drop_label_helper_article_dict(label_to_drop, gold_frame_article_id_dict, hat_frame_article_id_dict_ls[0])
            gold_frame_article_id_dict_ls[0] = data_dict['gold_frame_article_id_dict']
            hat_frame_article_id_dict_ls[0]  = data_dict['hat_frame_article_id_dict']

            
        
        overlap_dict = self.calculate_pairwise_overlap(gold_frame_article_id_dict, hat_frame_article_id_dict_ls)
        hat_include_master = overlap_dict['hat_include_master']
        gold_include_master = overlap_dict['gold_include_master']
        
        self.calculate_article_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict_ls[0], hat_frame_article_id_dict=hat_frame_article_id_dict_ls[0], 
                                        hat_include_master=hat_include_master, gold_include_master=gold_include_master,
                                        output_dir_path=segment_level_qualitative_result_output_dir_path, info=info_seg)
        segment_level_precision_a=self.article_metric_dict['avg']['precision'] #segment level using(article ids) precision
        segment_level_recall_a=self.article_metric_dict['avg']['recall']
        self.calculate_article_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict_ls[0], hat_frame_article_id_dict=hat_frame_article_id_dict_ls[1], 
                                        hat_include_master=hat_include_master, gold_include_master=gold_include_master,
                                        output_dir_path=segment_level_qualitative_result_output_dir_path, info=info_seg)
        segment_level_precision_b=self.article_metric_dict['avg']['precision'] #segment level using(article ids) precision
        segment_level_recall_b=self.article_metric_dict['avg']['recall']
        
        # assert safe_equal(segment_level_precision_a, segment_level_precision_b), \
        #     f"Precision mismatch: {segment_level_precision_a} != {segment_level_precision_b}"

        # assert safe_equal(segment_level_recall_a, segment_level_recall_b), \
        #     f"Recall mismatch: {segment_level_recall_a} != {segment_level_recall_b}"
        print('👍 Pairwise segment precision file1:', segment_level_precision_a)
        print('👍 Pairwise segment recall file1:', segment_level_recall_a)
        print('👍 Pairwise segment precision file2:', segment_level_precision_b)
        print('👍 Pairwise segment recall file2:', segment_level_recall_b)
        
    def calculate_all_metrics(self, gold_frame_article_id_dict, hat_frame_article_id_dict, hat_frame_dict, gold_frame_dict, article_ids_seen, cos_sim_thresh=0.5,
                                all_seen=True, info_frame=False, info_seg=False, frame_level_qualitative_result_path=None, segment_level_qualitative_result_output_dir_path=None, label_to_drop=None):
        '''
        hat_frame_article_id_dict: hat_frame as k and set of article ids as v
        gold_frame_article_id_dict: k is gold labels and v is set of article ids
        gold_frame_dict: k is gold label, v is list of segments labeled with this gold label
        hat_frame_dict: k is hat frame, v is a list of segments labeled with this hat frame
        '''
        data_dict = self.drop_label_helper(label_to_drop, gold_frame_article_id_dict, hat_frame_article_id_dict, hat_frame_dict, gold_frame_dict)
        gold_frame_article_id_dict = data_dict['gold_frame_article_id_dict']
        hat_frame_article_id_dict  = data_dict['hat_frame_article_id_dict']
        hat_frame_dict = data_dict['hat_frame_dict']
        gold_frame_dict = data_dict['gold_frame_dict']
        
        self.calculate_precision_recall(gold_frames=list(gold_frame_dict.keys()), hat_frames=list(hat_frame_dict.keys()), frame_level_qualitative_result_path=frame_level_qualitative_result_path, info=info_frame, cos_sim_thresh=cos_sim_thresh)
        self.calculate_article_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict, hat_frame_article_id_dict=hat_frame_article_id_dict, article_ids_seen=article_ids_seen, all_seen=all_seen, output_dir_path=segment_level_qualitative_result_output_dir_path, info=info_seg, cos_sim_thresh=cos_sim_thresh)
        try: self.silhouette_score=self.calculate_cluster_metrics(frame_seg_dict=hat_frame_dict)  
        except Exception as e:   
            self.silhouette_score=None
            print('[ERROR]',e)
            
        # frame level precision, frame level recall, segment level using(article ids) ['precision'], segment level using(article ids)['recall'], hat clusters with segments silhouette_score
        #return self.precision, self.recall, self.article_metric_dict['avg']['precision'], self.article_metric_dict['avg']['recall'], self.silhouette_score
        return {
                    "frame_level_precision": self.precision, # frame level precision
                    #"frame_level_precision_unique": self.precision_unique,
                    "frame_level_recall": self.recall, #frame level recall
                    #"frame_level_recall_unique": self.recall_unique,
                    "segment_level_precision": self.article_metric_dict['avg']['precision'], #segment level using(article ids) precision
                    "segment_level_recall": self.article_metric_dict['avg']['recall'], # segment level using(article ids) recall
                    "hat_silhouette_score": self.silhouette_score, #  hat clusters with segments silhouette_score
                }
    
import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
import numpy as np
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

    def calculate_article_metrics(self, gold_frame_article_id_dict, hat_frame_article_id_dict, article_ids_seen, cos_sim_thresh=0.5):
        # hat_frame_article_id_dict: hat_frame as k and set of article ids as v
        # gold_frame_article_id_dict: k is gold labels and v is set of article ids
        # sim_dict: key is hat frame : value is tuple of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        
        # Trim gold_frame_article_id_dict to article_ids that have been seen by model
        new_gold_frame_article_id_dict={ gold_frame : set() for gold_frame in gold_frame_article_id_dict.keys()}
        for gold_frame, article_ids in gold_frame_article_id_dict.items():
            for article_id in list(article_ids):
                if article_id in article_ids_seen:
                    new_gold_frame_article_id_dict[gold_frame].add(article_id)
        gold_frame_article_id_dict=new_gold_frame_article_id_dict
        
        # Calculate
        if self.sim_dict is None:
            self.sim_dict=self.get_sim_frame(gold_frames=list(gold_frame_article_id_dict.keys()), hat_frames=list(hat_frame_article_id_dict.keys()))
        total_hat=0
        for art_ids in hat_frame_article_id_dict.values():
            for id in art_ids: total_hat+=1
        metrics_dict={} #key is hat frame, v is a dict of metrics
        not_in_label_hframes=[] # list of not in gold_label hat frames
        precisions=[]
        recalls=[]
        f1s=[]
        for hat_frame, tup in self.sim_dict.items():
            cos_sim=tup[1]
            gold_frame=tup[0]
            if cos_sim > cos_sim_thresh:
                hat_art_ids=set(hat_frame_article_id_dict[hat_frame])
                gold_art_ids=set(gold_frame_article_id_dict[gold_frame])
                hat_dict={}
                hat_dict['true_positive'] = len(hat_art_ids & gold_art_ids)

                # Items in hat but not in gold (false positives)
                hat_dict['false_positive'] = len(hat_art_ids - gold_art_ids)

                # Items in gold but not in hat (false negatives)
                hat_dict['false_negative'] = len(gold_art_ids - hat_art_ids)
                hat_dict['precision'] = hat_dict['true_positive'] / (hat_dict['true_positive']+hat_dict['false_positive'])
                hat_dict['recall'] = hat_dict['true_positive'] / (hat_dict['true_positive'] + hat_dict['false_negative'])
                hat_dict['f1']=self.compute_f1(precision=hat_dict['precision'], recall=hat_dict['recall'])
                precisions.append(hat_dict['precision'])
                recalls.append(hat_dict['recall'])
                f1s.append(hat_dict['f1'])
                metrics_dict[hat_frame] = hat_dict
            else:
                not_in_label_hframes.append(hat_frame)
        if len(precisions)>0: avg_precision=sum(precisions)/len(precisions)
        else: avg_precision=float('nan')
        if len(recalls)>0: avg_recall=sum(recalls)/len(recalls)
        else: avg_recall=float('nan')
        if len(f1s)>0: avg_f1=sum(f1s)/len(f1s)
        else: avg_f1=float('nan')
        metrics_dict['avg']={'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}    
        self.article_metric_dict = metrics_dict
        
        
    def get_sim_frame(self, gold_frames, hat_frames, reverse=False):
        '''
        input: 
        gold_frames: data type is list, list of gold frames
        hat_frames: data type is list, list of predicted frames
        output:
        sim_dict: keys are all hat frames, values are pairs of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        
        for each hat_frame in hat_frames: find the most similar gold_frame and the cos sim score
        '''

        # Get embeddings for both lists
        hat_frames_embeddings = self.embedding_model.encode(hat_frames, convert_to_tensor=True, device='cuda')  # Shape: (len(hat_frames), embedding_dim)
        gold_frames_embeddings = self.embedding_model.encode(gold_frames, convert_to_tensor=True, device='cuda')  # Shape: (len(gold_frames), embedding_dim)

        # Normalize embeddings for cosine similarity
        hat_frames_embeddings = F.normalize(hat_frames_embeddings, p=2, dim=1)
        gold_frames_embeddings = F.normalize(gold_frames_embeddings, p=2, dim=1)

        if reverse:
            # Compute cosine similarity between each gold_frame and all hat_frames
            cos_sim_matrix = torch.matmul(gold_frames_embeddings, hat_frames_embeddings.T)  # Shape: (len(gold_frames), len(hat_frames))

            # Find the most similar hat_frame for each gold_frame
            sim_dict = {}
            for i, gold_frame in enumerate(gold_frames):
                best_match_idx = torch.argmax(cos_sim_matrix[i]).item()
                most_similar_hat_frame = hat_frames[best_match_idx]
                cos_sim_score = cos_sim_matrix[i, best_match_idx].item()
                sim_dict[gold_frame] = (most_similar_hat_frame, cos_sim_score)
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
    
    def calculate_precision_recall(self,gold_frames, hat_frames, cos_sim_thresh=0.5):
        '''
        embedding_model: the model use to encode frames/labels to vectors
        gold_frames: data type is list, list of gold frames
        hat_frames: data type is list, list of predicted frames
        '''
        
        # sim_dict: key is hat frame : value is tuple of (most similar gold frame for key, cos sim score between hat frame and most sim gold frame)
        self.sim_dict=self.get_sim_frame(gold_frames=gold_frames, hat_frames=hat_frames)
        self.sim_dict_for_gold_frame=self.get_sim_frame(gold_frames=gold_frames, hat_frames=hat_frames, reverse=True)
        
        self.true_positives_dict={}
        self.false_positives=[]
        true_positives = []
        #predicted_labels = [] # labels are keys of sim_dict, predicted labels are the most similar gold frames for each label
        for hat_frame, pair in self.sim_dict.items():
            cos_sim = pair[1]
            most_sim_gold_frame = pair[0]
            #predicted_labels.append(most_sim_gold_frame)
            if cos_sim > cos_sim_thresh:
                true_positives.append(most_sim_gold_frame)
                if most_sim_gold_frame in self.true_positives_dict:
                    self.true_positives_dict[most_sim_gold_frame].append(hat_frame)
                else: self.true_positives_dict[most_sim_gold_frame] = [hat_frame]
            else: self.false_positives.append(hat_frame)
    

        # deduplicated_true_positives = set(true_positives)
        # self.false_negatives = set(gold_frames) - deduplicated_true_positives
        # true_positives_num = len(deduplicated_true_positives)
        # false_negatives_num = len(gold_frames) - true_positives_num
        # false_positives_num = len(hat_frames) - len(true_positives)
        # self.precision = true_positives_num / (true_positives_num + false_positives_num)
        # self.recall = true_positives_num / (true_positives_num + false_negatives_num) # same as true_positives_num / len(gold_frames)
        
        self.precision_allow_duplication = len(true_positives) / len(hat_frames)
        self.precision_unique = set(true_positives) / len(hat_frames)

        
    def calculate_all_metrics(self, gold_frame_article_id_dict, hat_frame_article_id_dict, hat_frame_dict, gold_frame_dict, article_ids_seen):
        '''
        hat_frame_article_id_dict: hat_frame as k and set of article ids as v
        gold_frame_article_id_dict: k is gold labels and v is set of article ids
        gold_frame_dict: k is gold label, v is list of segments labeled with this gold label
        hat_frame_dict: k is hat frame, v is a list of segments labeled with this hat frame
        '''
        self.calculate_precision_recall(gold_frames=list(gold_frame_dict.keys()), hat_frames=list(hat_frame_dict.keys()))
        self.calculate_article_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict, hat_frame_article_id_dict=hat_frame_article_id_dict, article_ids_seen=article_ids_seen)
        try: self.silhouette_score=self.calculate_cluster_metrics(frame_seg_dict=hat_frame_dict)  
        except Exception as e:   
            self.silhouette_score=None
            print('[ERROR]',e)

        

        # frame level precision, frame level recall, segment level using(article ids) ['precision'], segment level using(article ids)['recall'], hat clusters with segments silhouette_score
        #return self.precision, self.recall, self.article_metric_dict['avg']['precision'], self.article_metric_dict['avg']['recall'], self.silhouette_score
        return {
                    "frame_level_precision": self.precision, # frame level precision
                    "frame_level_recall": self.recall, #frame level recall
                    'frame_level_f1': self.f1,
                    "segment_level_precision": self.article_metric_dict['avg']['precision'], #segment level using(article ids) precision
                    "segment_level_recall": self.article_metric_dict['avg']['recall'], # segment level using(article ids) recall
                    'segment_level_f1': self.article_metric_dict['avg']['f1'],
                    "hat_silhouette_score": self.silhouette_score, #  hat clusters with segments silhouette_score
                }
    
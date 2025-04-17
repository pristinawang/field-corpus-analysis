from metrics import *
from goldlabels import get_gold_labels
from sentence_transformers import SentenceTransformer
import json

def extract(json_path):
    # Open and load the JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    hat_frame_dict={}
    hat_frame_art_id_dict={}
    # Loop through the top-level keys
    for article_id, doc_content in data.items():
        for annotation in doc_content["LLM_Annotation"]:
            sentence=annotation["sentence"]
            labels=annotation["label"]
            for label in labels:
                if label in hat_frame_dict: hat_frame_dict[label].add(sentence)
                else: hat_frame_dict[label]={sentence}
                if label in hat_frame_art_id_dict: hat_frame_art_id_dict[label].add(article_id)
                else: hat_frame_art_id_dict[label]={article_id}
    return {
        "hat_frame_dict": hat_frame_dict,
        "hat_frame_article_id_dict": hat_frame_art_id_dict
    }
        
        
        

def run_metrics(embedding_model, file_paths, gold_silhouette_score, gold_frame_article_id_dict, gold_frame_dict):
    
    for file_path in file_paths:
        extractions=extract(json_path=file_path)   
        hat_frame_dict=extractions['hat_frame_dict']
        hat_frame_article_id_dict=extractions['hat_frame_article_id_dict']
        article_ids_seen=set()
        for ar_ids in hat_frame_article_id_dict.values():
            article_ids_seen.update(ar_ids)
        print('Seen article ids:', len(article_ids_seen))
        metrics=Metrics(embedding_model=embedding_model)
        metrics_dict=metrics.calculate_all_metrics(gold_frame_article_id_dict=gold_frame_article_id_dict, hat_frame_article_id_dict=hat_frame_article_id_dict, hat_frame_dict=hat_frame_dict, gold_frame_dict=gold_frame_dict, article_ids_seen=article_ids_seen)

        metrics_to_log = {
            "frame_level_precision": metrics_dict["frame_level_precision"],
            "frame_level_recall": metrics_dict["frame_level_recall"],
            "frame_level_f1":metrics_dict["frame_level_f1"],
            "segment_level_precision": metrics_dict["segment_level_precision"],
            "segment_level_recall": metrics_dict["segment_level_recall"],
            "segment_level_f1": metrics_dict["segment_level_f1"],
            "hat_silhouette_score": metrics_dict["hat_silhouette_score"],
            "gold_silhouette_score": gold_silhouette_score,
        }
        print('File path:', file_path, "frame_level_precision:", metrics_dict["frame_level_precision"], "frame_level_recall:", metrics_dict["frame_level_recall"],"frame_level_f1:", metrics_dict["frame_level_f1"], "segment_level_precision:", metrics_dict["segment_level_precision"],
                "segment_level_recall:", metrics_dict["segment_level_recall"],"segment_level_f1:", metrics_dict["segment_level_f1"],"hat_silhouette_score:", metrics_dict["hat_silhouette_score"],"gold_silhouette_score:", gold_silhouette_score)
    
def main():
    ## Config
    data='media'
    abs_file_paths=[
        '/data/afield6/oida_results/clustering/mediacorpus/gpt-4o-mini/base-model_llama/d26dcf98-edd7-11ef-bc0f-7cc25542b4b4.json',
        '/data/afield6/oida_results/clustering/mediacorpus/gpt-4o-mini/base-model_gpt-4o-mini/e3b2c3ce-f45f-11ef-920d-f402709bdab1.json',
        '/data/afield6/oida_results/clustering/mediacorpus/topicgpt/mediacorpus-assignment_reformat.json'
    ]
    
    ## Code starts
    
    embedding_model_id='all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(embedding_model_id).to('cuda')
    
    
    if data=='media':
        gold_frame_dict, gold_frame_arid_dict, _=get_gold_labels() #self.gold_frame_arid_dict: key is code; value is set of article ids
        metrics=Metrics(embedding_model=embedding_model)
        #gold_silhouette_score = metrics.calculate_cluster_metrics(frame_seg_dict=gold_frame_dict)
        gold_silhouette_score=-0.005740656
    else: raise ValueError("Data", data, "is not supported yet")
    run_metrics(embedding_model=embedding_model, file_paths=abs_file_paths, gold_silhouette_score=gold_silhouette_score, gold_frame_article_id_dict=gold_frame_arid_dict, gold_frame_dict=gold_frame_dict)
    

if __name__ == "__main__":
    main()
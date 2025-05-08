from metrics import *
from goldlabels import get_data_and_labels
from sentence_transformers import SentenceTransformer
import json, csv
from datetime import datetime

def extract(json_path):
    # Open and load the JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    hat_frame_dict={}
    hat_frame_art_id_dict={}
    all_art_ids=[]
    withlabel_art_ids=[]
    # Loop through the top-level keys
    for article_id, doc_content in data.items():
        article_id=str(article_id)
        all_art_ids.append(article_id)
        for annotation in doc_content["LLM_Annotation"]:
            sentence=annotation["sentence"]
            labels=annotation["label"]
            for i,label in enumerate(labels):
                withlabel_art_ids.append(article_id)
                if label in hat_frame_dict: hat_frame_dict[label].add(sentence)
                else: hat_frame_dict[label]={sentence}
                if label in hat_frame_art_id_dict: hat_frame_art_id_dict[label].add(article_id)
                else: hat_frame_art_id_dict[label]={article_id}
    print('Result file #art ids-all: len(all_art_ids) is',len(all_art_ids), 'len(set(all_art_ids)) is', len(set(all_art_ids)))
    print('Result file #art ids-with label: len(withlabel_art_ids) is',len(withlabel_art_ids), 'len(set(withlabel_art_ids)) is', len(set(withlabel_art_ids)))
    print('No label article ids:',list(set(all_art_ids)-set(withlabel_art_ids))[:5])
    return {
        "hat_frame_dict": hat_frame_dict,
        "hat_frame_article_id_dict": hat_frame_art_id_dict
    }
        
        
# def drop_label_helper(label_to_drop, label_dict_data):
#     label_dicts = [label_dict for label_dict in label_dict_data.values()]
#     new_data={}
#     for dict_name,label_dict in label_dict_data.items():
#         if not(label_to_drop in label_dict):
#             ValueError("label_to_drop is not in dict")
#         del label_dict[label_to_drop]
#         print(dict_name)
#         print(label_dict.keys())
#         new_data[dict_name] = label_dict
#     return new_data

def run_metrics(embedding_model, file_paths, gold_silhouette_score, gold_frame_article_id_dict, gold_frame_dict, dir_path, drop_label=None):
    metrics_path = os.path.join(dir_path, f"metrics_result.csv")
    csv_ls=[]
    for file_path in file_paths:
        print('--------------------------------------')
        print('📝 File path:', file_path)
        extractions=extract(json_path=file_path)   
        hat_frame_dict=extractions['hat_frame_dict']
        hat_frame_article_id_dict=extractions['hat_frame_article_id_dict']
        article_ids_seen=set()
        for ar_ids in hat_frame_article_id_dict.values():
            article_ids_seen.update(ar_ids)
        
        # Safety check
        ex_g_id = list(list(gold_frame_article_id_dict.values())[0])[0]
        ex_h_id = list(list(hat_frame_article_id_dict.values())[0])[0]
        assert isinstance(ex_g_id, str) and isinstance(ex_h_id, str), "Both gold_article_id and hat_article_id must be strings"
        art_ids_gold=set()
        for ar_ids in gold_frame_article_id_dict.values():
            art_ids_gold.update(ar_ids)
        print('Number of art ids with label(we use this set to calculate metrics):', len(article_ids_seen))
        print('Number of art ids in Gold:', len(art_ids_gold))
        print('Number of art ids in Gold not in Hat:', len(art_ids_gold-article_ids_seen))
        print('Example of art ids in Gold not in Hat:', list(art_ids_gold-article_ids_seen)[:5])
        print('Number of art ids in Hat not in Gold:', len(article_ids_seen-art_ids_gold))
        print('Example of art ids in Hat not in Gold:',list(article_ids_seen-art_ids_gold)[:5])
        
        # dataset = {
        #     'gold_frame_dict':gold_frame_dict,
        #     'gold_frame_article_id_dict':gold_frame_article_id_dict,
        #     'hat_frame_dict':hat_frame_dict,
        #     'hat_frame_article_id_dict': hat_frame_article_id_dict
        # }
        # if not(drop_label is None): dataset = drop_label_helper(drop_label, dataset)
        # output paths
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        subdir_path = os.path.join(dir_path, file_id)
        os.makedirs(subdir_path, exist_ok=True)
        # Create unique heatmap path
        cossim_path = os.path.join(subdir_path, f"frame_cossim.png")
        
        metrics=Metrics(embedding_model=embedding_model)
        metrics_dict=metrics.calculate_all_metrics(label_to_drop=drop_label, gold_frame_article_id_dict=gold_frame_article_id_dict, hat_frame_article_id_dict=hat_frame_article_id_dict, hat_frame_dict=hat_frame_dict, gold_frame_dict=gold_frame_dict, 
                                                    article_ids_seen=article_ids_seen, all_seen=True, frame_level_qualitative_result_path=cossim_path, segment_level_qualitative_result_output_dir_path=subdir_path,info_frame=True)
        
        metrics_to_log = {
            "file_path": file_path,
            "frame_level_precision": metrics_dict["frame_level_precision"],
            #"frame_level_precision_unique": metrics_dict["frame_level_precision_unique"],
            "frame_level_recall": metrics_dict["frame_level_recall"],
            #"frame_level_recall_unique": metrics_dict["frame_level_recall_unique"],
            "segment_level_precision": metrics_dict["segment_level_precision"],
            "segment_level_recall": metrics_dict["segment_level_recall"],
            "hat_silhouette_score": metrics_dict["hat_silhouette_score"],
            "gold_silhouette_score": gold_silhouette_score,
        }     
        csv_ls.append(metrics_to_log)
        # print('File path:', file_path, "frame_level_precision:", metrics_dict["frame_level_precision"],"frame_level_recall:", metrics_dict["frame_level_recall"],
        #         "segment_level_precision:", metrics_dict["segment_level_precision"], "segment_level_recall:", metrics_dict["segment_level_recall"],"hat_silhouette_score:", metrics_dict["hat_silhouette_score"],"gold_silhouette_score:", gold_silhouette_score)
        # print('--------------------------------------')
    os.makedirs(dir_path, exist_ok=True)
    # Write to CSV
    with open(metrics_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_ls[0].keys())
        writer.writeheader()
        writer.writerows(csv_ls)
    print('✅ Metrics csv file for all',len(file_paths),'files saved to', metrics_path)
def main():
    ## Config
    #data='val_e'
    data='val_e'
    media_file_paths=[
        '/data/afield6/oida_results/clustering/mediacorpus/gpt-4o-mini/base-model_llama/d26dcf98-edd7-11ef-bc0f-7cc25542b4b4.json',
        '/data/afield6/oida_results/clustering/mediacorpus/gpt-4o-mini/base-model_gpt-4o-mini/e3b2c3ce-f45f-11ef-920d-f402709bdab1.json',
        '/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/out/20250423124601/full_annotation_20250423124601.json',
        '/data/afield6/oida_results/clustering/mediacorpus/topicgpt/mediacorpus-assignment_reformat.json'
    ]
    astro_file_paths=[
        '/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/out/20250423222622/full_annotation_20250423222622.json',
        '/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/out/20250423223328/full_annotation_20250423223328.json',
        '/data/afield6/oida_results/clustering/astrobot/gpt-4o-mini/base-model_llama/astrobot_clustering.json',
        '/data/afield6/oida_results/clustering/astrobot/lloom/astrobot_lloom_clustering.json',
        '/data/afield6/oida_results/clustering/astrobot/topicgpt/astrobot_assignment_clustering.json'
    ]
    emo_file_paths=[
        '/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/out/20250423225107/full_annotation_20250423225107.json',
        '/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/out/20250423225800/full_annotation_20250423225800.json',
        '/data/afield6/oida_results/clustering/emotions/gpt-4o-mini/base-model_llama/c8265bdc-102c-11f0-a98f-7cc25542b43a_clustering.json',
        '/data/afield6/oida_results/clustering/emotions/topicgpt/emotions_assignment_clustering.json',
        '/data/afield6/oida_results/clustering/emotions/lloom/emotions_lloom_clustering.json'
    ]
    val_file_paths=[
        '/home/mzhong8/data_afield6/oida_results/clustering/values/topicgpt/values_assignment_clustering.json',
        '/home/mzhong8/data_afield6/oida_results/clustering/values/lloom/values_lloom_clustering.json',
        '/home/mzhong8/data_afield6/oida_results/clustering/values/81168dd0-2854-11f0-9e18-b49691db36ac_reformat/clustering/values/gpt-4o-mini/base-model_llama/81168dd0-2854-11f0-9e18-b49691db36ac_clustering.json',
        '/home/mzhong8/data_afield6/oida_results/clustering/values/81168dd0-2854-11f0-9e18-b49691db36ac_reformat/clustering/values/gpt-4o/base-model_llama/81168dd0-2854-11f0-9e18-b49691db36ac_clustering.json',
        '/home/mzhong8/data_afield6/oida_results/clustering/values/a1efbbca-284b-11f0-a331-b49691db36ac_reformat/clustering/values/gpt-4o-mini/base-model_llama/a1efbbca-284b-11f0-a331-b49691db36ac_clustering.json',
        '/home/mzhong8/data_afield6/oida_results/clustering/values/a1efbbca-284b-11f0-a331-b49691db36ac_reformat/clustering/values/gpt-4o/base-model_llama/a1efbbca-284b-11f0-a331-b49691db36ac_clustering.json',
        '/home/pwang71/pwang71/field/corpora_analysis/iter_codebook/out/20250424125046/full_annotation_20250424125046.json'
    ]
    val_topicgpt = ['/home/mzhong8/data_afield6/oida_results/clustering/values/topicgpt/values_assignment_clustering.json']
    astro_topicgpt = ['/data/afield6/oida_results/clustering/astrobot/topicgpt/astrobot_assignment_clustering.json']
    
    topicgpt = True
    abs_file_paths=val_topicgpt
    ## Code starts
    # Job id
    now = datetime.now()
    job_id = now.strftime("%Y%m%d%H%M%S")
    dir_path = os.path.join('./out/metrics', job_id)
    print('-------output dir path--------')
    print(dir_path)
    print('------------------------------')

    embedding_model_id='all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(embedding_model_id).to('cuda')
    
    gold_frame_dict, gold_frame_arid_dict, _=get_data_and_labels(data_type=data)
    if data=='media':
        #metrics=Metrics(embedding_model=embedding_model)
        #gold_silhouette_score = metrics.calculate_cluster_metrics(frame_seg_dict=gold_frame_dict)
        gold_silhouette_score=-0.005740656
    elif data=='astro' or data=='emo' or data=='val_e':
        metrics=Metrics(embedding_model=embedding_model)
        gold_silhouette_score = metrics.calculate_cluster_metrics(frame_seg_dict=gold_frame_dict)
    else: raise ValueError("Data", data, "is not supported yet")
    if not(topicgpt):
        drop_label=None
    else:
        if data=='media': drop_label = 'Political'
        elif data=='astro': drop_label = 'knowledge seeking: specific factual'
        elif data=='emo': drop_label = 'Responsive bots: voice assistants'
        elif data=='val_e': drop_label = 'Performance'
        else: raise ValueError("Data", data, "is not supported yet")
    run_metrics(embedding_model=embedding_model, file_paths=abs_file_paths, gold_silhouette_score=gold_silhouette_score, 
                gold_frame_article_id_dict=gold_frame_arid_dict, gold_frame_dict=gold_frame_dict, dir_path=dir_path, drop_label=drop_label)
    

if __name__ == "__main__":
    main()
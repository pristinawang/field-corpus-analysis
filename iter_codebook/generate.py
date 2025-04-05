from openai import OpenAI
from helper import *
import random
import numpy as np
import re, ast
from datetime import datetime
import json, os
from sentence_transformers import SentenceTransformer

import sys


# Get the parent directory (where corpora_analysis is located)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# print(parent_dir)
# Add it to sys.path
sys.path.append(parent_dir)
# Print sys.path to confirm
# print("Updated sys.path:")
# print("\n".join(sys.path))
from corpora_analysis.goldlabels import get_gold_labels
from corpora_analysis.metrics import *

def save_results(func):
    def wrapper(self, *args, **kwargs):  # Accept 'self'
        ## Save stats, codes, codebook
        ## codes in csv
        ## codebook in json
        ## stats in json
        iter_num = kwargs.get("iter_num", 0)
        # dir_path='./out/'+self.job_id+'/'
        # stats_file_name=dir_path+'stats_'+self.job_id+'.csv'
        # codes_file_name=dir_path+'codes_'+self.job_id+'.csv'
        # codebook_file_name=dir_path+'codebook_'+self.job_id+'.json'
        
        dir_path = os.path.join(self.output_dir, self.job_id)
        stats_file_name = os.path.join(dir_path, f'stats_{self.job_id}.csv')
        codes_file_name = os.path.join(dir_path, f'codes_{self.job_id}.csv')
        os.makedirs(dir_path, exist_ok=True)
        
        print(f"Calling {func.__name__}...")

        result = func(self, *args, **kwargs)
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"
        #code_max_sim_dict=get_max_label_sim(embedding_model=self.embedding_model, labels=list(self.codebook_dict.keys()))
        save_codes_csv(func_name=func.__name__, iter_num=iter_num, codebook_dict=self.codebook_dict,file_full_path=codes_file_name)
        if self.metrics_bool:
            hat_frame_article_dict=self.get_hat_frame_articleID_dict()
            metrics_to_csv(func_name=func.__name__, iter_num=iter_num,metrics_bool=self.metrics_bool, embedding_model=self.embedding_model, gold_frame_article_id_dict=self.gold_frame_arid_dict, hat_frame_article_id_dict=hat_frame_article_dict, hat_frame_dict=self.get_frame_segs_dict(self.codebook_dict),gold_frame_dict=self.gold_frame_dict,gold_silhouette_score=self.gold_silhouette_score,file_full_path=stats_file_name)
            
    
        print(f"Finished {func.__name__}")
        return result
    return wrapper


    
class IterCoder:
    def __init__(self, job_id, dataset, embedding_model, num_seg_first_batch, batch_size,stop_thresh, update_loop_num, output_dir, seg_low_thresh=2, drop_freq_thresh=2,random_seed=42, delimiter='\n\n', splitter_article="PRIMARY", metrics_bool=False):
        '''
        dataset: dict; keys are article ids, vals are articles
        '''
        self.job_id=job_id # job_id is the file name of the output saved
        self.dataset=dataset
        self.embedding_model=embedding_model
        self.output_dir=output_dir
        self.random_seed=random_seed
        random.seed(self.random_seed)
        self.drop_freq_thresh=drop_freq_thresh
        self.seg_low_thresh=seg_low_thresh
        self.num_seg_first_batch=num_seg_first_batch
        self.batch_size=batch_size
        self.stop_thresh=stop_thresh # determine the threshold for stopping; we stop when the number of labels with only 1 segment is below or equal stop_thresh
        self.update_loop_num=update_loop_num # the number of update loops we run. after that, we only run merge and drop module
        self.article_ids=dataset.keys()
        self.articles=dataset.values()
        self.articles_arr=np.array(self.articles)
        self.delimiter= delimiter
        # self.segments=[segment for article in self.articles for segment in article.split(splitter_article, 1)[-1].strip().split(self.delimiter)]
        # self.segment_to_article = {
        #     segment: article_id
        #     for article_id, article in zip(self.article_ids, self.articles)
        #     for segment in article.split(splitter_article, 1)[-1].strip().split(self.delimiter)
        # }
        self.segments=[]
        self.segment_art_id_dict={} # key is segment(str form), value is article id
        for article_id, article in zip(self.article_ids, self.articles):
            for segment in article.split(splitter_article, 1)[-1].strip().split(self.delimiter):
                self.segments.append(segment)
                self.segment_art_id_dict[segment]=article_id
        
        
        self.remaining_seg_ids=list(range(len(self.segments)))
        self.segments_arr=np.array(self.segments)
        max_iter_num=round((len(self.segments)-self.num_seg_first_batch)/self.batch_size)+1 #max number of iteration including first batch
        self.iteration_progress=range(1,max_iter_num) # from 1 to (max_iter_num-1): after first batch
        self.codebook=[]
        self.codebook_dict={} #key is code; value is a list of segment ids
        self.low_seg_freq_dict={} # key is code; value is the number of times this code has 1 segment
        self.format_error_count=0
        self.deduplication_error_log={} #key: iteration(int); value: log(str); the log itself shows deleted ids in format 1-N instead of 0-N-1
        self.merge_error_log={}
        self.update_codebook_log={} #key: iteration(int); value: log(str)
        self.clean_label_log={} #key: iteration(int); value: log(str)
        self.__eval_log={} #key: iteration(int); value: stats(dict); stats{key:[added num of codes, ended with num of codes]; value:[int,int]}
        self.metrics_bool=metrics_bool
        if self.metrics_bool:
            self.gold_frame_dict, self.gold_frame_arid_dict=get_gold_labels() #self.gold_frame_arid_dict: key is code; value is set of article ids
            gold_metrics=Metrics(embedding_model=self.embedding_model)
            #self.gold_silhouette_score = gold_metrics.calculate_cluster_metrics(frame_seg_dict=self.gold_frame_dict)
            self.gold_silhouette_score=-0.005740656
        #self.num_seg_per_article=len(self.articles[0].split(self.delimiter))
    
    def output_final_results(self):
        dir_path = os.path.join(self.output_dir, self.job_id)
        stats_file_name = os.path.join(dir_path, f'stats_{self.job_id}.csv')
        
        plot_stats(dir_path=dir_path, csv_path=stats_file_name)
        self.output_final_annotations()
    
    def output_final_annotations(self):
        
        
        ## Save final annotations
        ## get a list of article ids that are picked
        article_ids=set()
        codes=[]
        seg_id_ls=[]
        for code, seg_ids in self.codebook_dict.items():
            for id in seg_ids:
                article_id = self.segment_art_id_dict[self.segments[id]]
                article_ids.add(article_id)
                codes.append(code)
                seg_id_ls.append(id)
        article_ids=list(article_ids)
        
        ## get a dict: segment_label_dict: segment(str) is key, value is list of labels
        segment_label_dict={}
        for code, seg_id in zip(codes, seg_id_ls):
            if self.segments[seg_id] in segment_label_dict: segment_label_dict[self.segments[seg_id]].append(code)
            else: segment_label_dict[self.segments[seg_id]]=[code]
        
        ## initialize blank result dict
        result_dict={}
        for article_id in article_ids:
            result_dict[article_id]={"LLM_Annotation":[], "Text":self.dataset[article_id]}
        
        # fill in result dict
        for segment, label_ls in segment_label_dict.items():
            article_id=self.segment_art_id_dict[segment]
            result_dict[article_id]["LLM_Annotation"].append({"label":label_ls, "sentence": segment})

        dir_path = os.path.join(self.output_dir, self.job_id)
        result_file_path = os.path.join(dir_path, f'final_annotation_{self.job_id}.json')
        os.makedirs(dir_path, exist_ok=True)

        # Write the result_dict to the JSON file
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print("🎉 Final annotations has been saved to", result_file_path)

    def get_hat_frame_articleID_dict(self):
        '''
        key is code
        value is a set of the associated segment's article id
        '''
        hat_frame_dict={}
        for code, seg_ids in self.codebook_dict.items():
            ids=set()
            for id in seg_ids:
                article_id = self.segment_art_id_dict[self.segments[id]]
                ids.add(article_id)
            hat_frame_dict[code]=ids
        return hat_frame_dict
            
    def stop_condition(self):
        
        if self.batch_size > len(self.remaining_seg_ids):
            print('Stopping criteria-> batch size:',self.batch_size,'> # remaining seg:', len(self.remaining_seg_ids))
            return True
        else:
            count_of_one_seg_labels=self.get_count_of_one_seg_labels()
            if count_of_one_seg_labels <= self.stop_thresh:
                print('# Count of labels with only 1 segment is below or equal to stop_thresh', '# Count of 1 seg labels:',count_of_one_seg_labels, 'thresh:', self.stop_thresh)
                return True
            else: return False
    
    def get_count_of_one_seg_labels(self):
        count_of_one_seg_labels=0
        for code, seg_ids in self.codebook_dict.items():
            if len(seg_ids)<=1: count_of_one_seg_labels+=1
        return count_of_one_seg_labels
        
    
    def get_eval_added_num_codes(self, iter_num):
        return self.__eval_log[iter_num]['added number of codes']
    def get_eval_total_num_codes(self, iter_num):
        return self.__eval_log[iter_num]['total number of codes']
    def get_eval_log_keys(self):
        return list(self.__eval_log.keys())
    def save_eval_log(self, iter_num, added_num_codes, total_num_codes):
        self.__eval_log[iter_num]={'added number of codes':added_num_codes, 'total number of codes': total_num_codes}
    
    def get_pure_ans(self, text):
        # Regex pattern to extract text after 'Ans:'
        #match = re.search(r'Ans:\s*(.*)', text)
        match = re.search(r'Ans:\s*(.*)', text, re.DOTALL)

        if match:
            result = match.group(1)  # Extract the matched group
            return result.strip(' ') #return result
        else:
            self.format_error_count+=1
            raise ValueError("Error: 'Ans:' not found in the string.")
    
    def get_list_fromtxtlist(self, txt_list):
        try:
            int_list=ast.literal_eval(txt_list)
            return int_list
        except:
            self.format_error_count+=1
            raise ValueError("Error: txt list cannot be converted to int list")

    
    def get_segments(self,n):
        '''
        n: number of segments
        '''
        if n > len(self.remaining_seg_ids):
            raise ValueError(f"Requested {n} segments, but only {len(self.remaining_seg_ids)} segments are available.")
        random.shuffle(self.remaining_seg_ids)
        selecting_ids=np.array(self.remaining_seg_ids[:n])
        self.remaining_seg_ids=self.remaining_seg_ids[n:] 
        selected_segments=list(self.segments_arr[selecting_ids])
        return selected_segments, list(selecting_ids)
    
    def get_frame_segs_dict(self, dict):
        '''
        codebook_dict uses segment ids as values
        convert that into a dict that has hat_frame as k, list of segments(str) as values
        '''
        frame_segs_dict={}

        for code,segment_ids in dict.items():
            frame_segs_dict[code] = [self.segments[id] for id in segment_ids]
        return frame_segs_dict
    
    @save_results
    def generate_initial_codebook(self, debug, iter_num=0, prompt_path='./prompt/first_batch_prompt.txt'):
        segments, segment_ids =self.get_segments(self.num_seg_first_batch)
        prompt=txt_to_string(prompt_path)
        full_prompts = list(map(lambda s: prompt + s, segments))
        generated_texts=generate_responses_batch(full_prompts)
        if debug:
            print('----OG text-----')
            print(generated_texts)
        assert len(segment_ids) == len(generated_texts), f"Length mismatch: {len(segment_ids)} vs {len(generated_texts)}"

        ## First Batch Codebook
        for i, text in enumerate(generated_texts):
            try:
                label=self.get_pure_ans(text=text)
                if label not in self.codebook_dict:self.codebook_dict[label]=[segment_ids[i]]
                else: 
                    if not(segment_ids[i] in self.codebook_dict[label] ): 
                        self.codebook_dict[label].append(segment_ids[i])
            except:
                print("generate_initial_codebook(): might be RE error, 'Ans:' not found in the string."+"; Text:"+text)
        self.codebook=list(self.codebook_dict.keys())
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"
        if debug:
            print("-------Codebook----------")
            print(self.codebook)
    
    def flatten(self, list):
        '''
        id: starts at 1
        '''
        strout=''
        for i,item in enumerate(list):
            strout=strout+' '+str(i+1)+'. '+item+','
        return strout.strip(',')
    
    def refine_codes(self):
        pass
    
    # def deduplication(self, round_log, prompt_path='./prompt/deduplication_prompt.txt'):
    #     prompt=txt_to_string(prompt_path)
    #     full_prompt=prompt+self.flatten(list=self.codebook)
    #     print('----full prompt-----')
    #     print(full_prompt)
    #     generated_text=generate_response_single(prompt=full_prompt)
    #     print('---Dupl gen txt----')
    #     print(generated_text)
    #     print('--------pure ans--------')

    #     txt_list=self.get_pure_ans(text=generated_text)
    #     print(txt_list)
    
    def flatten_dict(self, dict):
        strr=''
        i=1
        dict_list=list(dict.items())
        random.shuffle(dict_list)
        for code,segment_ids in dict_list:
            
            strr = strr+'\n\n'+str(i)+'. '+code
            seg_str = '\n\n-> '.join([self.segments[id] for id in segment_ids])
            strr = strr + '\n\n-> ' + seg_str
            i+=1
        return strr.strip('\n')

    def deduplication(self, iter_num, debug, prompt_path='./prompt/deduplication_prompt.txt'):
        '''
        iter_num: int
        It does deduplication and deletion. Cannot separate two since prompt doesn't work.
        Simply delete the codes that are identified as duplications
        recode() after deletion of codes
        '''
        deduplication_log=''
        prompt=txt_to_string(prompt_path)
        full_prompt=prompt+self.flatten(list=self.codebook)
        if debug:
            print('----Dupl full prompt-----')
            print(full_prompt)
        generated_text=generate_response_single(prompt=full_prompt)
        if True:
            print('---Dupl gen txt----')
            print(generated_text)
            print('--------pure ans--------')
        try:
            txt_list=self.get_pure_ans(text=generated_text)
            if debug: print(txt_list)
            deleting_ids=self.get_list_fromtxtlist(txt_list=txt_list)
            if debug: print(deleting_ids)
        except:
            deduplication_log=deduplication_log+'deduplication failed at get_pure_ans or get_list_fromtxtlist'
            self.deduplication_error_log[iter_num]=deduplication_log
            return
        seg_ids_deleted=[] 
        #codes_deleted=[]
        for id in deleting_ids:
            id=id-1
            key=self.codebook[id]
            if key in self.codebook_dict:  # Prevent KeyError
                seg_ids_deleted=seg_ids_deleted+self.codebook_dict[key]
                #codes_deleted.append(key)
                del self.codebook_dict[key]
        if len(self.codebook)==len(self.codebook_dict.keys()):
            deduplication_log=deduplication_log+'no deduplication happened. deleting_ids: '+', '.join(map(str, list(np.array(deleting_ids)))) #changed to match codebook log format(id:1-N instead of 0-N-1) join(map(str, list(np.array(deleting_ids)-1)))
            self.deduplication_error_log[iter_num]=deduplication_log
            return

        deduplication_log=deduplication_log+'Deduplication Success\n-> OG Codebook: '+', '.join(self.codebook)+'\n-> Deleted ids: '+', '.join(map(str, list(np.array(deleting_ids)))) # changed to match codebook log format(id:1-N instead of 0-N-1) join(map(str, list(np.array(deleting_ids))))
        self.codebook=list(self.codebook_dict.keys())
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"
        deduplication_log=self.recode(deduplication_log=deduplication_log, seg_ids_deleted=seg_ids_deleted, debug=debug)
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"
        self.deduplication_error_log[iter_num]=deduplication_log

    def recode(self, deduplication_log, seg_ids_deleted, debug, prompt_path='./prompt/recode_prompt.txt'):
        '''
        Reassign the segments of duplicated codes to the updated codebook
        '''
        ## Maybe batch prompt; each for one answer; Ans: id or N/A
        prompt=txt_to_string(prompt_path)
        codes_deleted_str='\nCodebook:'+self.flatten(list=self.codebook)
        segments=[]
        for seg_id in seg_ids_deleted:
            seg=self.segments[seg_id]
            segments.append(seg)
        full_prompts = list(map(lambda s: prompt + codes_deleted_str +'\nSegment:\n'+ s, segments))
        generated_texts=generate_responses_batch(full_prompts)
        if debug:
            print('------------Recode prompt-----------')
            for p in full_prompts: print(p)
            print('------------Recode Gen txt---------')
            for t in generated_texts: print(t)
        for i, text in enumerate(generated_texts):
            seg=segments[i]
            try:
                id=int(self.get_pure_ans(text=text))-1
            except:
                deduplication_log=deduplication_log+'\nRecode not happening: N/A or int() error\n-> generated txt: '+text+'\n-> Segment: '+seg
                continue
            if id>=0 and id < len(self.codebook):
                code=self.codebook[id]
                if code in self.codebook_dict: 
                    if not(seg_ids_deleted[i] in self.codebook_dict[code]):
                        self.codebook_dict[code].append(seg_ids_deleted[i])
                        assert self.segments[seg_ids_deleted[i]]==seg, f"Stored segment id doesn't match segment text: {self.segments[seg_ids_deleted[i]]} vs {seg}"
                        deduplication_log=deduplication_log+'\nRecode success\n-> id: '+str(id)+'\n-> generated txt:\n'+text+'\n-> Segment: '+seg
                    else:
                        assert self.segments[seg_ids_deleted[i]]==seg, f"Stored segment id doesn't match segment text: {self.segments[seg_ids_deleted[i]]} vs {seg}"            
                        deduplication_log=deduplication_log+'\nRecode Duplication: segment already assigned to this code\n-> id: '+str(id)+'\n-> generated txt:\n'+text+'\n-> Segment: '+seg
                else: 
                    deduplication_log=deduplication_log+'\nRecode failed: *this should not happen, sth is wrong* chosen code not in codebook-> generated txt: '+text+'; Segment: '+seg
            else: 
                deduplication_log=deduplication_log+'\nRecode failed: code id not in range-> generated txt: '+text+'; Segment: '+seg
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"
        return deduplication_log

    def old_merge(self, debug, round_log, seg_bool, prompt_path='./prompt/merge_prompt.txt'):
        '''
        merge() adds new code and original codes' segments but doesn't remove old code to "ensure that no information is lost when a merge choice isn't good"
        effect is -> one segment can be assigned with multiple codes
        '''
        prompt=txt_to_string(prompt_path)
        if seg_bool:full_prompt=prompt+'\n'+self.flatten_dict(dict=self.codebook_dict)
        else: full_prompt=prompt+self.flatten(list=self.codebook)
        if debug:
            print('----full prompt-----')
            print(full_prompt)
        generated_text=generate_response_single(prompt=full_prompt)
        if debug:
            print('---Merge gen txt----')
            print(generated_text)
        try:
            json_txt=self.get_pure_ans(text=generated_text)
            if debug:
                print('------Pure Ans-------')
                print(json_txt)
            merge_dict=self.get_merge_pure_json(json_txt=json_txt)
            if debug:
                print('------Pure JSON-------')
                print(merge_dict)
        except:
            self.merge_error_log.append(round_log+': Format error, merge failed at get_pure_ans() or get_merge_pure_json()\n-> Gen txt:\n'+json_txt)
            return
        self.merge_error_log.append(round_log+':')
        for merge in merge_dict['merges']:
            if merge['merged_into'] in self.codebook_dict:
                self.merge_error_log[-1] = self.merge_error_log[-1] +' Failure -> new code(merged_into) already exists -> merged into: '+merge['merged_into']
                continue
            merged_into=merge['merged_into']
            og_codes=merge['original_codes']
            self.merge_error_log[-1] = self.merge_error_log[-1] +'\n merged into: '+merged_into
            new_seg_ids=[]
            for og_code in og_codes:
                if og_code in self.codebook_dict: new_seg_ids=new_seg_ids + self.codebook_dict[og_code]
            if len(new_seg_ids)>0: 
                self.codebook_dict[merged_into]=new_seg_ids
                self.merge_error_log[-1] = self.merge_error_log[-1] + '\n-> Successfully added new code: '+merged_into+'; OG codes(at lease one was added, not sure how many): '+', '.join(og_codes)
            else: 
                self.merge_error_log[-1] = self.merge_error_log[-1] + '\n-> Failed to add new code: '+merged_into+'; OG codes(None was added): '+', '.join(og_codes)
        self.codebook=list(self.codebook_dict.keys())
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"

    def merge_merge_prompt(self, debug, round_log, seg_bool, prompt_path='./prompt/merge_prompt_seg.txt'):
        '''
        merge() merges original_codes into merged_into
        old merge function using "merging prompt"
        '''
        prompt=txt_to_string(prompt_path)
        if seg_bool:full_prompt=prompt+'\n'+self.flatten_dict(dict=self.codebook_dict)
        else: full_prompt=prompt+self.flatten(list=self.codebook)
        if debug:
            print('----full prompt-----')
            print(full_prompt)
        generated_text=generate_response_single(prompt=full_prompt)
        if debug:
            print('---Merge gen txt----')
            print(generated_text)
        try:
            json_txt=self.get_pure_ans(text=generated_text)
            if debug:
                print('------Pure Ans-------')
                print(json_txt)
            merge_dict=self.get_merge_pure_json(json_txt=json_txt)
            if debug:
                print('------Pure JSON-------')
                print(merge_dict)
        except:
            self.merge_error_log.append(round_log+': Format error, merge failed at get_pure_ans() or get_merge_pure_json()\n-> Gen txt:\n'+json_txt)
            return
        self.merge_error_log.append(round_log+':')
        for merge in merge_dict['merges']:
            if merge['merged_into'] in self.codebook_dict:
                self.merge_error_log[-1] = self.merge_error_log[-1] +'\n Failure -> new code(merged_into) already exists -> merged into: '+merge['merged_into']
                continue
            merged_into=merge['merged_into']
            og_codes=merge['original_codes']
            og_codes_existing=[]
            self.merge_error_log[-1] = self.merge_error_log[-1] +'\n merged into: '+merged_into
            new_seg_ids=[]
            for og_code in og_codes:
                if og_code in self.codebook_dict: 
                    new_seg_ids=new_seg_ids + self.codebook_dict[og_code]
                    og_codes_existing.append(og_code)
            if len(new_seg_ids)>0: 
                self.codebook_dict[merged_into]=new_seg_ids
                for og_code_existing in og_codes_existing: del self.codebook_dict[og_code_existing]
                self.merge_error_log[-1] = self.merge_error_log[-1] + '\n-> Successfully added new code: '+merged_into+'; OG codes(all existing og code in this list have been merged and old code deleted, might not be all): '+', '.join(og_codes_existing)
            else: 
                self.merge_error_log[-1] = self.merge_error_log[-1] + '\n-> Failed to add new code: '+merged_into+'; OG codes(None was added): '+', '.join(og_codes)
        self.codebook=list(self.codebook_dict.keys())
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"

    @save_results
    def drop(self, iter_num):
        '''
        iter_num: int
        record the number of times a code has below seg_low_thresh number of segments
        remain under seg_low_thresh for greater than drop_freq_thresh amount of times will be dropped
        '''
        ## Record
        for code, segs in self.codebook_dict.items():
            if len(segs)<self.seg_low_thresh:
                if code in self.low_seg_freq_dict: self.low_seg_freq_dict[code]+=1
                else: self.low_seg_freq_dict[code]=1
        ## Drop
        # Who to be dropped
        to_be_dropped=[]
        for code in self.low_seg_freq_dict.keys():
            if self.low_seg_freq_dict[code]>self.drop_freq_thresh:
                to_be_dropped.append(code)
        # Dropping
        for code in to_be_dropped:
            del self.codebook_dict[code]
            del self.low_seg_freq_dict[code]
        self.codebook=list(self.codebook_dict.keys())
        print('Num of codes dropped', len(to_be_dropped))

    @save_results
    def merge(self, debug, iter_num, seg_bool, prompt_path='./prompt/merge_prompt_seg.txt'):
        '''
        iter_num: int
        merge() clusters original_codes into high_level_code
        '''
        merge_log=''
        prompt=txt_to_string(prompt_path)
        if seg_bool:full_prompt=prompt+'\n'+self.flatten_dict(dict=self.codebook_dict)
        else: full_prompt=prompt+self.flatten(list=self.codebook)
        if debug:
            print('----full prompt-----')
            print(full_prompt)
        generated_text=generate_response_single(prompt=full_prompt)
        if debug:
            print('---Merge gen txt----')
            print(generated_text)
        try:
            json_txt=self.get_pure_ans(text=generated_text)
            if debug:
                print('------Pure Ans-------')
                print(json_txt)
            merge_dict=self.get_merge_pure_json(json_txt=json_txt)
            if debug:
                print('------Pure JSON-------')
                print(merge_dict)
        except:
            merge_log=merge_log+str(iter_num)+': Format error, merge failed at get_pure_ans() or get_merge_pure_json()\n-> Gen txt:\n'+generated_text
            self.merge_error_log[iter_num]=merge_log
            return
        merge_log=merge_log+str(iter_num)+':'
        for merge in merge_dict['clusters']:
            if merge['high_level_code'] in self.codebook_dict:
                merge_log=merge_log +'\n Failure -> new code(high_level_code) already exists -> high_level_code: '+merge['high_level_code']
                continue
            merged_into=merge['high_level_code']
            og_codes=merge['original_codes']
            og_codes_existing=[]
            merge_log=merge_log +'\n high_level_code: '+merged_into
            new_seg_ids=[]
            for og_code in og_codes:
                if og_code in self.codebook_dict: 
                    new_seg_ids=new_seg_ids + self.codebook_dict[og_code]
                    og_codes_existing.append(og_code)
            if len(new_seg_ids)>0: 
                self.codebook_dict[merged_into]=new_seg_ids
                for og_code_existing in og_codes_existing: 
                    if og_code_existing in self.codebook_dict:
                        del self.codebook_dict[og_code_existing]
                merge_log=merge_log  + '\n-> Successfully added new code: '+merged_into+'; OG codes(all existing og code in this list have been merged and old code deleted, might not be all): '+', '.join(og_codes_existing)
            else: 
                merge_log=merge_log + '\n-> Failed to add new code: '+merged_into+'; OG codes(None was added): '+', '.join(og_codes)
        self.merge_error_log[iter_num]=merge_log
        self.codebook=list(self.codebook_dict.keys())
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"

    def clean_label_names(self, iter_num, verbose, prompt_path='./prompt/clean_prompt.txt'):
        prompt=txt_to_string(prompt_path)
        old_labels=[]
        log=''
        for code,segment_ids in list(self.codebook_dict.items()):
            
            
            # seg_str = '\n\n'.join([self.segments[id] for id in segment_ids])
            # full_prompt=prompt+code+'\n\nText:'+seg_str
            full_prompt=prompt+code
            generated_text=generate_response_single(prompt=full_prompt)
            try:
                label=self.get_pure_ans(text=generated_text)
            except:
                self.clean_label_log[iter_num]="get_pure_ans() RE error, 'Ans:' not found in the string."+"; Text:"+generated_text
                continue
            if not(label in self.codebook_dict):
                self.codebook_dict[label]=segment_ids
                old_labels.append(code)
                log=log+code+' --> '+label+'\n'
        for old_label in old_labels:
            del self.codebook_dict[old_label]
        self.clean_label_log[iter_num]=log
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"

    @save_results 
    def update_codebook(self, iter_num, codebook_bool, verbose, prompt_path='./prompt/update_codebook_prompt.txt'):
        '''
        iter_num: int
        codebook_bool: including codebook in prompt or not
        '''
        if self.batch_size > len(self.remaining_seg_ids): return
        
        
        update_log=''
        segments, segment_ids = self.get_segments(n=self.batch_size)
        prompt=txt_to_string(prompt_path)
        for i, segment in enumerate(segments):
            segment_id=segment_ids[i]
            codebook_str=self.flatten(list=self.codebook)
            if codebook_bool: full_prompt=prompt+codebook_str+'\n\nSegment: '+segment
            else: full_prompt=prompt+'\n\nSegment: '+segment
            generated_text=generate_response_single(prompt=full_prompt)
            try:
                label=self.get_pure_ans(text=generated_text)
            except:
                update_log=update_log+'\n'+'Seg #'+str(i)+": get_pure_ans() RE error, 'Ans:' not found in the string."+"; Text:"+generated_text
                continue
            if label not in self.codebook_dict: 
                self.codebook_dict[label]=[segment_id] 
                update_log=update_log+'\n'+'Seg #'+str(i)+": Added new code, New code:"+label
            else:
                if not(segment_id in self.codebook_dict[label]): 
                    self.codebook_dict[label].append(segment_id)
                    update_log=update_log+'\n'+'Seg #'+str(i)+": Segment added to Existing code: Code:"+label+'\nSegment: '+segment
                else:
                    update_log=update_log+'\n'+'Seg #'+str(i)+": Segment already added to this code: Code:"+label+'\nSegment: '+segment
        
        self.update_codebook_log[iter_num]=update_log
        self.save_eval_log(iter_num=iter_num, added_num_codes=len(self.codebook_dict)- len(self.codebook), total_num_codes=len(self.codebook_dict)) 
        self.codebook=list(self.codebook_dict.keys())
        if verbose: print('Iter:',iter_num,'; Added # Codes:',self.get_eval_added_num_codes(iter_num=iter_num), '; Total # Codes:', self.get_eval_total_num_codes(iter_num=iter_num))
        assert len(self.codebook) == len(self.codebook_dict), f"Length mismatch: {len(self.codebook)} vs {len(self.codebook_dict)}"
        
    def get_merge_pure_json_old_merge_prompt(self,json_txt):
        """
        OLD MERGE PROMPT
        Parses LLM output into a dictionary and validates required keys and data types.

        Args:
            llm_response (str): The raw JSON response from the LLM.

        Returns:
            dict: A structured dictionary if validation passes.

        Raises:
            ValueError: If the response format is invalid or missing required keys.
        """
        try:

            # Parse JSON
            response_data = json.loads(json_txt)

            # Validate top-level structure
            if not isinstance(response_data, dict):
                raise ValueError("Error: The response is not a valid dictionary.")

            # Ensure "merges" key exists and is a list
            if "merges" not in response_data:
                raise ValueError("Error: Missing required key 'merges'.")
            if not isinstance(response_data["merges"], list):
                raise ValueError("Error: 'merges' must be a list.")

            # Validate structure of each merge entry
            for merge in response_data["merges"]:
                if not isinstance(merge, dict):
                    raise ValueError("Error: Each item in 'merges' must be a dictionary.")
                
                # Required keys
                required_keys = ["merged_into", "original_codes", "justification"]
                for key in required_keys:
                    if key not in merge:
                        raise ValueError(f"Error: Missing key '{key}' in a merge entry.")
                
                # Validate data types
                if not isinstance(merge["merged_into"], str):
                    raise ValueError("Error: 'merged_into' must be a string.")
                if not isinstance(merge["original_codes"], list) or not all(isinstance(code, str) for code in merge["original_codes"]):
                    raise ValueError("Error: 'original_codes' must be a list of strings.")
                if not isinstance(merge["justification"], str):
                    raise ValueError("Error: 'justification' must be a string.")

            return response_data  # Return parsed and validated dictionary

        except (json.JSONDecodeError, ValueError) as e:
            self.format_error_count+=1
            raise ValueError(f"Invalid LLM response format(get_merge_pure_json() failed): {e}")
    
    def get_merge_pure_json(self,json_txt):
        """
        Parses LLM output into a dictionary and validates required keys and data types.

        Args:
            llm_response (str): The raw JSON response from the LLM.

        Returns:
            dict: A structured dictionary if validation passes.

        Raises:
            ValueError: If the response format is invalid or missing required keys.
        """
        try:

            # Parse JSON
            response_data = json.loads(json_txt)

            # Validate top-level structure
            if not isinstance(response_data, dict):
                raise ValueError("Error: The response is not a valid dictionary.")

            # Ensure "merges" key exists and is a list
            if "clusters" not in response_data:
                raise ValueError("Error: Missing required key 'clusters'.")
            if not isinstance(response_data["clusters"], list):
                raise ValueError("Error: 'clusters' must be a list.")

            # Validate structure of each merge entry
            for merge in response_data["clusters"]:
                if not isinstance(merge, dict):
                    raise ValueError("Error: Each item in 'clusters' must be a dictionary.")
                
                # Required keys
                required_keys = ["high_level_code", "original_codes", "justification"]
                for key in required_keys:
                    if key not in merge:
                        raise ValueError(f"Error: Missing key '{key}' in a cluster entry.")
                
                # Validate data types
                if not isinstance(merge["high_level_code"], str):
                    raise ValueError("Error: 'high_level_code' must be a string.")
                if not isinstance(merge["original_codes"], list) or not all(isinstance(code, str) for code in merge["original_codes"]):
                    raise ValueError("Error: 'original_codes' must be a list of strings.")
                if not isinstance(merge["justification"], str):
                    raise ValueError("Error: 'justification' must be a string.")

            return response_data  # Return parsed and validated dictionary

        except (json.JSONDecodeError, ValueError) as e:
            self.format_error_count+=1
            raise ValueError(f"Invalid LLM response format(get_merge_pure_json() failed): {e}")
    
    


        
    def run(self, verbose):
        self.generate_initial_codebook(debug=False)
        if verbose:
            print('----Iter: 0, Og Codebook-------')
            print(self.flatten_dict(dict=self.codebook_dict))
        # self.deduplication(iter_num=0, debug=False)
        # print('----depl log------')
        # print('#'+str(0),self.deduplication_error_log[0])
        # print('---New Codebook after Dupl----')
        # print(self.flatten_dict(dict=self.codebook_dict))
        # print('---------Merge no seg-------------')
        # self.merge(round_log='Round #1', seg_bool=False)
        
        self.merge(iter_num=0,debug=False, seg_bool=True, prompt_path='./prompt/merge_prompt_seg.txt')
        if verbose:
            print('---------Merge with seg----------')
            print('----merge log------')
            print('#'+str(0),self.merge_error_log[0])
            print('---New Codebook after Merge----')
            print(self.flatten_dict(dict=self.codebook_dict))
        self.drop(iter_num=0)
        if verbose:
            print('------Update Loop Starts----------')
        for iter_num in self.iteration_progress:
            if iter_num <= self.update_loop_num:
                self.update_codebook(iter_num=iter_num, codebook_bool=False, verbose=True)
                if verbose:
                    print('-------#'+str(iter_num)+' Updated Codebook---------')
                    print(self.flatten_dict(dict=self.codebook_dict))
                    print('----#'+str(iter_num)+' update log------')
                    print('#'+str(iter_num),self.update_codebook_log[iter_num])
            
            # self.deduplication(iter_num=iter_num, debug=False) ## TODO deleting all codes that are the same, not even leaving one
            # print('----#'+str(iter_num)+' depl log------')
            # print('#'+str(iter_num),self.deduplication_error_log[iter_num])
            # print('---New Codebook after Dupl----')
            # print(self.flatten_dict(dict=self.codebook_dict))
            self.merge(iter_num=iter_num, debug=False, seg_bool=True, prompt_path='./prompt/merge_prompt_seg.txt')
            if verbose:
                print('----#'+str(iter_num)+' merge log------')
                print('#'+str(iter_num),self.merge_error_log[iter_num])
                print('---New Codebook after Merge----')
                print(self.flatten_dict(dict=self.codebook_dict))
            self.drop(iter_num=iter_num)
            # self.clean_label_names(iter_num=iter_num, verbose=False)
            # print('----#'+str(iter_num)+' Clean Name log------')
            # print(self.clean_label_log[iter_num])

            if self.stop_condition(): 
                print('Iter ended #'+ str(iter_num))
                break 
        self.output_final_results()
        print('---> Format error?')
        print(self.format_error_count)
        
def main():
    # Job id
    now = datetime.now()
    job_id = now.strftime("%Y%m%d%H%M%S")
    print('--------job_id: output path-------')
    print('./out/'+job_id+'/')
    print('--------------------------------')
    ## Embedding Model
    embedding_model_id='all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(embedding_model_id).to('cuda')
    
    # Iterative coding
    article_dict=read_input_segments(info=True, info_exp=False) #article_dict: dict type; key: article id, value: article text
    small_dataset=dict(list(article_dict.items())[:50])
    all_dataset=dict(list(article_dict.items()))
    print('-------Double Check Input data info----------')
    print('Num of articles:', len(list(article_dict.keys())))
    gold_frame_dict, gold_frame_arid_dict=get_gold_labels()
    itercoder=IterCoder(job_id=job_id, dataset=all_dataset, embedding_model=embedding_model, output_dir='./out',num_seg_first_batch=32, batch_size=48, stop_thresh=0, delimiter='.\n\n', update_loop_num=20, metrics_bool=True)
    itercoder.run(verbose=False)
    
    
if __name__=='__main__':
    main()
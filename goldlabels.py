import json
def get_gold_labels():
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
    print('All art', len(allart), 'rel art', len(art))  
    return frame_dict, frame_arid_dict, article_dict # key is code; value is set of article ids



def getcodes(path='/data/afield6/afield6/moksh/media_frames_corpus/codes.json'):
    with open(path, 'r') as file:
        codes = json.load(file)
    return codes
if __name__=='__main__':
    get_gold_labels()
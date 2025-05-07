import json
import csv
import requests
from bs4 import BeautifulSoup

def process_data(path='/data/afield6/oida_data/processed/values/uplifted_values_full.json', 
                    output_path='/home/pwang71/pwang71/field/corpora_analysis/uplifted_values_full.csv'):
    full_path='/data/afield6/oida_data/processed/values/uplifted_values_full.json'
    expli_path='/data/afield6/oida_data/processed/values/uplifted_values_explicit.json'
    output_full_path='/home/pwang71/pwang71/field/corpora_analysis/uplifted_values_full.csv'
    output_expli_path='/home/pwang71/pwang71/field/corpora_analysis/uplifted_values_explicit.csv'
    output_quote_path='/home/pwang71/pwang71/field/corpora_analysis/uplifted_values_quote.csv'
    # Load JSON file
    with open(full_path, "r", encoding="utf-8") as f:
        data_full = json.load(f)
    with open(expli_path, "r", encoding="utf-8") as f:
        data_expli = json.load(f)
    
    q_id=1
    quote_id_dict={}
    # Create quote id
    for doc_id, doc_info in data_full.items():
        title=doc_info['title']
        url=doc_info['url']
        annotations=doc_info['annotation']
        for annotation in annotations:
            quotation=annotation['quotation']
            if not(quotation in quote_id_dict): 
                quote_id_dict[quotation]=q_id
                q_id+=1
    for doc_id, doc_info in data_expli.items():
        title=doc_info['title']
        url=doc_info['url']
        annotations=doc_info['annotation']
        for annotation in annotations:
            quotation=annotation['quotation']
            if not(quotation in quote_id_dict): 
                quote_id_dict[quotation]=q_id
                q_id+=1
    quote_id_ls=[]
    data_f_ls=[]
    data_e_ls=[]
    # Loop through each document
    for doc_id, doc_info in data_full.items():
        title=doc_info['title']
        url=doc_info['url']
        annotations=doc_info['annotation']
        for annotation in annotations:
            quotation=annotation['quotation']
            section=annotation['section']
            labels=annotation['labels']
            if labels is not None:
                data_dict={'quote_id':quote_id_dict[quotation], 'section':section, 'labels':labels, 'quotation':quotation,'title':title, 'url':url}
                data_f_ls.append(data_dict)
    with open(output_full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data_f_ls[0].keys())
        writer.writeheader()
        writer.writerows(data_f_ls)
        
    # Loop through each document
    for doc_id, doc_info in data_expli.items():
        title=doc_info['title']
        url=doc_info['url']
        annotations=doc_info['annotation']
        for annotation in annotations:
            quotation=annotation['quotation']
            section=annotation['section']
            labels=annotation['labels']
            if labels is not None:
                data_dict={'quote_id':quote_id_dict[quotation], 'section':section, 'labels':labels, 'quotation':quotation,'title':title, 'url':url}
                data_e_ls.append(data_dict)
    with open(output_expli_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data_f_ls[0].keys())
        writer.writeheader()
        writer.writerows(data_e_ls)
    
    for quote, q_id in quote_id_dict.items():
        data={'quote_id':q_id, 'quotation':quote}
        quote_id_ls.append(data)
    with open(output_quote_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=quote_id_ls[0].keys())
        writer.writeheader()
        writer.writerows(quote_id_ls)

def return_acl_paper_titles(year: str):
    '''
    input:
    year of acl main track papers to extract, str; Ex: '2024'
    
    output:
    list of titles of papers, list of str
    '''
    year_int=int(year)
    if year_int==2020:
        url='https://aclanthology.org/volumes/'+year+'.acl-main/'
    elif year_int<2020:
        raise ValueError(f"This year is not supported yet({year}).")
    else:
        url = "https://aclanthology.org/volumes/"+year+".acl-long/"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error while accessing the URL: {url}. The requested year probably doesn't exist ({year}).")
    soup = BeautifulSoup(response.content, "html.parser")

    # This finds all paper titles
    titles = []
    for strong in soup.find_all("strong"):
        a_tag = strong.find("a", class_="align-middle")
        if a_tag:
            titles.append(a_tag.text.strip())
    return titles[1:]

def crawl_acl(year:str, output_path='/home/pwang71/pwang71/field/corpora_analysis/out/created_data/acl.csv'):
    titles=return_acl_paper_titles(year=year)
    n = len(titles)
    failed_titles=[]
    data_ls=[]
    for i in range(1,n+1):
        paper_id='acl-long-'+year+'-'+str(i)
        title=titles[i-1]
        url='https://aclanthology.org/2024.acl-long.'+str(i)+'/'
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, "html.parser")
        abstract_div = soup.find("div", class_="card-body acl-abstract")

        if abstract_div:
            abstract_text = abstract_div.find("span").get_text(strip=True)
            data={'paper_id':paper_id, 'title':title, 'abstract':abstract_text}
            data_ls.append(data)
        else:
            failed_titles.append(title)
    if len(failed_titles)>0: print(failed_titles)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data_ls[0].keys())
        writer.writeheader()
        writer.writerows(data_ls)

    


if __name__=='__main__':
    crawl_acl('2024')

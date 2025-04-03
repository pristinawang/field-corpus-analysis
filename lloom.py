import os
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
import text_lloom.workbench as wb
import asyncio
os.environ["OPENAI_API_KEY"] = "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"

# We'll load data from an existing CSV
data_link = "/home/pwang71/pwang71/field/corpora_analysis/astro_emotions/media_moksh.csv"
#df = pd.read_csv(data_link)

try:
    # Try reading with utf-8 encoding
    df = pd.read_csv(data_link, encoding='utf-8')
except UnicodeDecodeError:
    # If utf-8 fails, try a different encoding
    df = pd.read_csv(data_link, encoding='latin1')
    
l = wb.lloom(
    df=df,
    text_col="frames",
    id_col="doc_id",  # Optional
)

params = l.auto_suggest_parameters(sample_size=None, target_n_concepts=10)
params['synth_n_concepts']=17
print('params:',params)

cur_seed = None  # Optionally replace with string
#await l.gen(seed=cur_seed,params=params) #, params=params

async def main():
    await l.gen(seed=cur_seed, params=params)  # Ensure this runs inside an async function

# Run the async function
asyncio.run(main())
try:
    l.summary()
    print(l.summary())
except:
    l.summary()

try:
    l.select()
    print(l.select())
except:
    l.select()
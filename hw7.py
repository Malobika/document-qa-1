import os
import re
import uuid
import json
from typing import List, Tuple

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# --- Chroma setup (sqlite on some hosts needs pysqlite3 shim)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import chromadb 
from chromadb.errors import IDAlreadyExistsError

CHROMA_PATH=r"./"
COLLECTION_NAME="news_chunks"


def build_vector_db(chunks,client):
    filenames= "f{os.listdir.path}.part{i}"
    IDs= os.listdir.

    #initialize a persistantdb client
    chroma_client=chromadb.PersistentClient(path=CHROMA_PATH)
    #define a collection that can store the collection and if the path is empty can create one 

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    #check is embeddings already there:
    if collections len>0:
        return collection

    #now we create embeddings 
    emb = client.embeddings.create( chunks, model="")
    #embeddings added to collection 
    collection.add(emb,filename,Ids)

    return collection

def chunk_data(filepath):
    #read csv save in pandas
    read_csv=pandas_df(filepath)
    #put data in each column
    dict1=[company_name,_,Date,Document,URL]= col[0],col[1],col[2],col[3],col[4]
    #each news item will be a chunk 
    dict1.add[col[URL-DATA]]

    for each_url in dict1[col(URL_DATA)]:
        chunk = covert_html_to_text(each_url)
        #use beautiful soup to parse through url and add the data

        for i, chunk in enumerate([chunk] ,start=1):
            emb=openai_client.embeddings.create(
                model="text-embedding-3-small"
                input=chunks
            ).data[0].embedding
            doc_ids= os.path.basename(path)
            meta=os.path.basename(path)
            #put it in try catch coz we dont know if collection exists
            collections.add[document=[chunks],emb=[emb],ids=[doc_ids],metadata=[meta]]
            #throw some exception
    st.success("Vector DB created ✅")
    return collections

def validateOpenAiApiclient():
    api_key = os.getenv("OPENAI_API_KEY")
    #IF API KEY VALID
    client= Openaikey(api_key=api_key)
    #make a function call"
    try:
        m=client.models.list()
        
    catch Exception:
    #key not valid
        return False
    return True 
def validateClaudeApiclient():
    api_key = os.getenv("Claude_API_KEY")
    #IF API KEY VALID
    client= Claudekey(api_key=api_key)
    #make a function call"
    try:
        m=client.models.list()
        
    catch Exception:
    #key not valid
        return False
    return True 

def chatbot():
    validateOpenAiApiclient()
    validateClaudeApiclient()
    st.chat_message({"role": "system","content":"This is a AI Chatbot to explore news. Please ask as many questions as you can"})
    if not messages:
        st.session_state.messages=[]
    prompt := st.chat_input()
    if prompt:
        





           
    






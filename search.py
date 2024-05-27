#write python code to read from a supabase table named oa1_aao from a db named o1db to do vector embedding search
#the code should read the table and do a vector search on the column named summary_embedding
#the code should return the top 5 rows that are closest to the input vector
#the code should return the rows in a list

import numpy as np
import pandas as pd
import requests
import json
import os

#function that takes a user query and converts it to a vector which is then used in the search function

def query_to_vector(query):
    #get user query
    query = query
    

def search(input_vector):
    #read the supabase url and key from the environment variables
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    #read the table name and db name
    table_name = "oa1_aao"
    db_name = "o1db"
    #read the column name for the vector embeddings
    column_name = "summary_embedding"
    #read the input vector
    input_vector = input_vector
    #create the url for the supabase api
    url = f"{url}/rest/v1/{db_name}/{table_name}?select=*&order={column_name},asc"
    #make a get request to the supabase api
    response = requests.get(url, headers={"apikey": key})
    #get the data from the response
    data = response.json()
    #create a list to store the rows
    rows = []
    #loop through the data and get the rows
    for row in data:
        rows.append(row)
    #return the rows
    return rows


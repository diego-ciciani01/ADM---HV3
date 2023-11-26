import numpy as np
from bs4 import BeautifulSoup
import requests
from nltk.stem import *
import nltk
from nltk.corpus import stopwords
import string 
import pandas as pd
from forex_python.converter import CurrencyRates
from collections import defaultdict
import heapq
from geopy.geocoders import Nominatim

# 1. Data collection
# 1.1 get the list of master's degree courses
def extract_masters(this_url):
    result_url=requests.get(this_url)
    result_soup=BeautifulSoup(result_url.text)
    result_links=result_soup.find_all('a',{'class':'courseLink'})
    result_list=[]
    for item in result_links:
        result_list.append((item['href'],item.text))
    return result_list

# 1.3 Parse downloaded pages
def extract_msc_page(file_path):
    course_info = {}
    with open(file_path, 'r', encoding='utf-8') as file:
      contents = []
      page_soup = BeautifulSoup(file, 'html.parser')
        
      course_containers = page_soup.find_all('div', class_='course-header')
      course_data_containers = page_soup.find_all('div', class_='course-data__container')
      coruse_link = page_soup.find_all('link', href=True)
      for course_container in course_containers:
            #Course Name
            name_links = course_container.find_all('h1', class_='course-header__course-title')
            if name_links:
                course_info['courseName'] = name_links[0].text.strip()
            else:
                course_info['courseName'] = ''

            #University Name
            university_links = course_container.find_all('a', class_='course-header__institution')
            if university_links:
                course_info['universityName'] = university_links[0].text.strip()
            else:
                  course_info['universityName'] = ''

            #Faculty Name (department)
            faculty_links = course_container.find_all('a', class_='course-header__department')
            if faculty_links:
                course_info['facultyName'] = faculty_links[0].text.strip()
            else:
              course_info['facultyName'] = ''

          #Full or Part Time
            full_time_links = course_container.find_all('a', class_='concealLink')
            fullTime = 'Part Time'
            for item in full_time_links:
              if item['href'] == '/masters-degrees/full-time/':
                    fullTime = 'Full Time'
                    break
            course_info['isItFullTime'] = fullTime

            #Description
            description = page_soup.find('div', id='Snippet')
            description_without_tags =  description.get_text()
            course_info['description'] = description_without_tags.replace('\n', '')

            #Start data
            start_data = page_soup.find('span', class_='key-info__start-date')
            course_info['start_data'] = start_data.text

            #Modality
            qualification_data = page_soup.find('span', class_="key-info__qualification")
            course_info['modality'] = qualification_data.text

            #Duration
            duration_data = page_soup.find('span', class_="key-info__duration")
            course_info['duration'] = duration_data.text

            #Fees
            fees_data = page_soup.find('div', class_="course-sections__fees")
            fees_without_tags = fees_data.get_text()
            course_info['fees'] = fees_without_tags.replace('\n', '')

              #get course data container
    for course_data_cointainer in course_data_containers:
        #Country
        country_links = course_data_cointainer.find_all('a', class_='course-data__country')
        if country_links:
            course_info['country'] = country_links[0].text.strip()
        else:
            course_info['country'] = ''
        
        #City
        city_links = course_data_cointainer.find_all('a', class_='course-data__city')
        if city_links:
            course_info['city'] = city_links[0].text.strip()
        else:
            course_info['city'] = ''
        
        #Administrator
        administrator_links = course_data_cointainer.find_all('a', class_='course-data__on-campus')
        if administrator_links:
            course_info['administrator'] = administrator_links[0].text.strip()
        else:
            course_info['administrator'] = 'On line'

    course_info["url"] = coruse_link[0]['href']
    
    contents.append(course_info)
    return contents


# 2. Search Engine
# 2.0.0 Preprocessing the text

# 1. stemming
stemmer = PorterStemmer()
# Define a function to handle stemming
def stem_description(description):
    return [stemmer.stem(word) for word in description.split(' ')]
    
# 2. Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a function to remove stopwords 
stop_words = set(stopwords.words('english'))
def remove_stopwords(words_list):
    return [word for word in words_list if word not in stop_words]
    
# 3. Remove punctuation
def remove_punctuation(words):
    return [word for word in words if word not in string.punctuation]

# 2.0.1 Preprocessing the fees column 

# function to convert the symbol to the letteral currency
symbol_to_name = {
    '$': 'USD',
    '€': 'EUR',
    '£': 'GBP',
    '¥': 'JPY',
    '₹': 'INR'}
def converti_simbolo_a_letterale(simbolo):
    return symbol_to_name.get(simbolo, simbolo)

c = CurrencyRates()
def convert_currency(amount, from_currency, to_currency):
    if from_currency == to_currency:
      return from_currency
    else :
      return c.convert(from_currency, to_currency, amount)
    
def convert_and_replace(x):
    if pd.notnull(x):
        costo = float(x['costo'].replace(',', ''))
        valuta = converti_simbolo_a_letterale(x['valuta']) if x['valuta'] else None
        
        if valuta:
            converted_amount = convert_currency(costo, valuta, 'EUR')
            return converted_amount
        else:
            # Se la valuta è None, restituisci il valore originale
            return costo
    else:
        return x


# 2.1.2 Execute the query
def query_preprocess(text):

    # remove anything not necessary as newlines and slashes
    text = text.replace("\\n", " ").replace("/", " ").replace("-", " ")

    # remove punctuation 
    for el in string.punctuation:
        text = text.replace(el, '')
    
    # split into list of words
    text = text.lower()
    text = text.split(" ")

    # stemming 
    stemmer = PorterStemmer()
    stem = [stemmer.stem(word) for word in text]

    # removing stopwords
    text = [word for word in stem if word not in stop_words]
    
    return text

#2.2.1 Conjunctive query
# create second inverted index
def create_second_inverted_index(inv_indx, vocabulary, tfidataset_data, feat):
    # feat is the column we're querying on 
    extended_inverted_index = defaultdict(list)
    
    # Iterate through each term in the inverted index
    for term_id, doc_indices in inv_indx.items():
        # Iterate through each document index for the current term
        
        for doc_index in doc_indices:
            # Get the TF-Idataset scores for the current document 
            word = vocabulary[vocabulary['term_id'] == term_id]['term'].values
         
            if  word[0] in tfidataset_data.columns: # check if the word is in the tfidataset
                    tfidataset_scores = tfidataset_data.loc[doc_index,word[0]]
            else:
                    continue
    
            # Append a tuple of (document_index, TF-Idataset scores) to the term's list in the extended inverted index
            extended_inverted_index[term_id].append((doc_index, tfidataset_scores))
    
    # Convert the extended inverted index defaultdict to a regular dictionary
    extended_inverted_index = dict(extended_inverted_index)
    
    # save the extended inverted dictionary in a txt file as before
    name_file = f'Extended Inverted Index {feat}.txt'
    with open(name_file, 'w') as file:
    
        for key, value in extended_inverted_index.items():
            file.write(f'{key}: {value}\n')
    file.close()

# read the inverted index from the file.
def read_inverted_index(feat):
    name_file = f'Extended Inverted Index {feat}.txt'
    file = open(name_file, "r")
    
    ext_inv_indx_feat = dict()
    txt = file.read().split("\n")
    
    for i in range(len(txt)-1):
        line = txt[i].replace(":", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "").split(" ")
        ext_inv_indx_feat[int(line[0])] = []
        for j in range(1, len(line)):
            if j%2 == 1:
                ext_inv_indx_feat[int(line[0])].append((int(line[j]), float(line[j+1])))
                
    file.close()
    return ext_inv_indx_feat

# 2.2.2 Execute the query 

# create a vector for the query
def create_vector_query(query, vocabulary, tfidataset_data):
    query_vec = np.zeros(vocabulary.shape[0]) # inizialize the vector
    for word in query:
        if word in tfidataset_data.columns:
            term_id = vocabulary[vocabulary['term'] == word]['term_id']
            query_vec[term_id] = 1.0
    return query_vec

# compute the cosine similarity
def a_cosine_similarity(query_vec, doc_arr):
    try:
        cos_sim =np.dot(query_vec, doc_arr) / (np.linalg.norm(doc_arr) * np.linalg.norm(query_vec))
        if cos_sim is None or np.isnan(cos_sim) or np.isinf(cos_sim):
            return 0.0  # Return zero if fails 
        return cos_sim  # Return the computed cosine similarity
    except Exception as e:
        print("Error in cosine similarity calculation:", e)
        return 0.0  # Return a value indicating failure
    

def execute_query(k, heap):
    top_k = heapq.nlargest(k, heap)    
    print(top_k)
    top_doc_k = []
    
    #fill the list of top_k_doc
    for score, doc in top_k:
        top_doc_k.append(doc)
    print(top_doc_k)
    return top_k, top_doc_k
    
# 4. Visualizing the most relevant MSc degrees
#Function to obtain the coordinates given the name of the university, the city and the country
def get_coordinates(universityName, city, country):

    full_address = f"{universityName}, {city}, {country}"
    small_address = f"{city}, {country}"
    
    geolocator = Nominatim(user_agent="location_finder")
    
    try:
        #Attempt to geocode with full address
        full_location = geolocator.geocode(full_address, timeout=10)
        if full_location and full_location.latitude is not None and full_location.longitude is not None:
            return full_location.latitude, full_location.longitude
        else:
            #If full address geocoding fails or returns null coordinates, try with city and country
            small_location = geolocator.geocode(small_address, timeout=10)
            if small_location:
                return small_location.latitude, small_location.longitude
            else:
                return None, None
    except Exception as e:
        print(f"Error during the geocodify: {e}")
        return None, None
    
# 5. Bonus point
# Function to check if any of the months of the starting date are present in the start date
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
current_month = 11
months_to_keep = months[current_month-4:current_month]
def filter_months(row):
    start_dates = row.split(', ')
    return any(month in start_dates for month in months_to_keep)
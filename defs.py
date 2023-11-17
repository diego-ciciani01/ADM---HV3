import numpy as np
from bs4 import BeautifulSoup
import requests
from nltk.stem import *
import nltk
from nltk.corpus import stopwords
import string 

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
      for course_container in course_containers:
            #Course Name
            name_links = course_container.find_all('h1', class_='course-header__course-title')
            if name_links:
                course_info['courseName'] = name_links[0].text.strip()
            else:
                course_info['courseName'] = ''
            
            # url ??
            

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
            fullTime = False
            for item in full_time_links:
              if item['href'] == '/masters-degrees/full-time/':
                    fullTime = True
                    break
            course_info['isItFullTime'] = fullTime

            description = page_soup.find('div', id='Snippet')
            description_without_tags =  description.get_text()
            course_info['description'] = description_without_tags.replace('\n', '')

            start_data = page_soup.find('span', class_='key-info__start-date')
            course_info['start_data'] = start_data.text

            qualification_data = page_soup.find('span', class_="key-info__qualification")
            course_info['modality'] = qualification_data.text

            duration_data = page_soup.find('span', class_="key-info__duration")
            course_info['duration'] = duration_data.text

            fees_data = page_soup.find('div', class_="course-sections__fees")
            fees_without_tags = fees_data.get_text()
            course_info['fees'] = fees_without_tags.replace('\n', '')

              #get course data container
    for course_data_cointainer in course_data_containers:

        country_links = course_data_cointainer.find_all('a', class_='course-data__country')
        if country_links:
            course_info['country'] = country_links[0].text.strip()
        else:
            course_info['country'] = ''

        city_links = course_data_cointainer.find_all('a', class_='course-data__city')
        if city_links:
            course_info['city'] = city_links[0].text.strip()
        else:
            course_info['city'] = ''

        administrator_links = course_data_cointainer.find_all('a', class_='course-data__on-campus')
        if administrator_links:
            course_info['administrator'] = administrator_links[0].text.strip()
        else:
            course_info['administrator'] = ''


      #aggiungere url della pagina nel datafeame
    contents.append(course_info)
    return contents


# 2. Search Engine
# 2.0.0 Preprocessing the text

# 1. stemming
stemmer = PorterStemmer()
# Define a function to handle stemming with checks for None values
def stem_description(description):
    if description is not None and isinstance(description, str):  # Check if description is not None and is a string
        return [stemmer.stem(word) for word in description.split(' ')]
    else:
        return []  # Return an empty list for None or non-string values
    
# 2. Remove stopwords
nltk.download('stopwords')

# Define a function to handle stopwords and stemming with checks for None values
def clean_description(description):
    lst_stopwords = stopwords.words('english')
    if description is not None and isinstance(description, str):  # Check if description is not None and is a string
        words = description.split(' ')
        return [stemmer.stem(word) for word in words if word not in lst_stopwords]
    else:
        return []  # Return an empty list for None or non-string values
    
# 3. Remove punctuation
def clean_description(description):
    lst_stopwords = stopwords.words('english')
    if description is not None and isinstance(description, str):  
        # Check if description is not None and is a string
        words = nltk.word_tokenize(description)
        cleaned_words = [stemmer.stem(word) for word in words if word not in lst_stopwords and word.isalnum()]
        return cleaned_words
    else:
        return []  # Return an empty list for None or non-string values
    

# 2.0.1 Preprocessing the fees column ????????????????????????????????????????????

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
    text = [stemmer.stem(word) for word in text]
    
    return text

# 2.2.2 Execute the query 
def a_cosine_similarity(query_vec, doc_arr):
    # Your implementation for cosine similarity calculation
    # Ensure to handle cases where the computation might result in None
    # If cosine similarity calculation fails, return a value that indicates failure (e.g., -1)
    # If successful, return the computed cosine similarity value
    try:
        cos_sim =np.dot(query_vec, doc_arr) / (np.linalg.norm(doc_arr) * np.linalg.norm(query_vec))
        if cos_sim is None or np.isnan(cos_sim) or np.isinf(cos_sim):
            return -1  # Return a value indicating failure
        return cos_sim  # Return the computed cosine similarity
    except Exception as e:
        print("Error in cosine similarity calculation:", e)
        return -1  # Return a value indicating failure
    

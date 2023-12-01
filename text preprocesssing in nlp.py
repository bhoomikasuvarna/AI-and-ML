#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import emoji
# pip install emoji
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select the relevant column
    ds = df.iloc[:, [1]]
    ds
    
    # Convert emojis to text
    ds['emoji_text'] = ds['comment_text'].apply(lambda x: emoji.demojize(x))
    

      # Remove digits
    ds['remove_digits'] = ds["emoji_text"].replace(to_replace=r'\d', value='', regex=True)
    
    
    # Convert to lowercase
    ds = ds.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
      # Appcontracotionly contractions
    ds["contraction"] = ds["remove_digits"].apply(lambda x: contractions.fix(str(x)))
    
    ds['url_remove'] = ds["contraction"].apply(lambda x: re.sub(r'http\S+', '', str(x)))
    
    
    # Remove non-word and non-whitespace characters
    ds["whitespace"] = ds["url_remove"].replace(to_replace=r'[^\w\s]', value='', regex=True)
    
  
    # Tokenization
    ds['Tokenized_Comment'] = ds['whitespace'].apply(lambda x: word_tokenize(str(x)))
    
# #     # Stopword Removal
    stop_words = set(stopwords.words('english'))
    ds['stopword_Comment'] = ds['Tokenized_Comment'].apply(lambda x: [word for word in x if word not in stop_words])
    
# #     # Stemming
    stemmer = PorterStemmer()
    ds['stemmed_Comment'] = ds['stopword_Comment'].apply(lambda x: [stemmer.stem(word) for word in x])
    
# #     # Lemmatization
    lemmatizer = WordNetLemmatizer()
    def lemmatize_tokens(tokens):
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    ds['lemmatized_Comment'] = ds['stopword_Comment'].apply(lemmatize_tokens)
    
    return ds

processed_data = preprocess_text_data("Call Me by Your Name.csv")
processed_data


# In[3]:


pip install emoji


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





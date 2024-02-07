#!/usr/bin/env python
# coding: utf-8

# In[16]:



import pandas as pd
import re
import string


# In[17]:



from sklearn.model_selection import train_test_split


# In[18]:



data=pd.read_csv('Call Me by Your Name.csv')
data


# In[19]:


import nltk
nltk.download('punkt')


# In[20]:


pip install emot


# In[21]:


pip install demoji


# In[22]:



import pandas as pd


# In[23]:


#from textblob import TextBlob
import emot
import nltk.data
import numpy as np
import re
import time
import string
import emoji
import demoji
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import TweetTokenizer
import regex
import nltk
from nltk.stem import SnowballStemmer
demoji.download_codes()
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import LinearSVC


# In[24]:


x=data['text']
y=data['task1']
data.head(5)


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0, test_size=0.25)


# In[26]:


data['task1'].value_counts()


# In[27]:



#emot_object = emot.core.emot()
ps =PorterStemmer()
lemmatiser = WordNetLemmatizer()
english_stopwords = stopwords.words('english')
exclude = set(string.punctuation)
def preprocess(text):
  #text=demoji.findall(df['Text'])
    text = contractions.fix(text.lower(), slang=True)
    text =re.sub("@ ?[A-Za-z0-9_]+", "", text)
    text= re.sub(r'\d+', '', text)
    text=re.sub(r'$', '', text)
    text= re.sub(r'’','', text )
    text=re.sub('<.*?>','',text)
    text=re.sub(r'http\S+', '', text)
  #text=emoji.demojize(text, delimiters=(" ", " "))
    text = ''.join(ch for ch in text if ch not in exclude)
    tokens = word_tokenize(text)
  #print("Tokens:", tokens)
    text = [t for t in tokens if t not in english_stopwords]
    text = " ".join(text)
    return text


# In[28]:



import emoji
#import demoji
#demoji.download_codes()
def emo(text):
    
    temp=emoji.demojize(text,delimiters=(" "," "))
    temp=temp.replace("_","  ")
    return temp


# In[29]:


data['clean_text']=data["text"].apply(lambda x:emo(x))
data["clean_text"]=data['clean_text'].apply(lambda X: preprocess(X))


# In[30]:



data


# In[31]:


import regex

def custom_analyzer(text):
    words = regex.findall(r'\w{2,}', text) # extract words of at least 2 letters
    for w in words:
        yield w


# In[32]:


tf_idf1 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
#applying tf idf to training data
X_train_tf1 = tf_idf1.fit_transform(data['clean_text'])
#applying tf idf to training data
X_train_tf1 = tf_idf1.transform(data['clean_text'])


# In[33]:


#transforming test data into tf-idf matrix
X_test_tf1 = tf_idf1.transform(data["clean_text"])


# In[34]:


from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train_tf1, data['task1'])


# In[35]:



y_pred = lsvc.predict(X_test_tf1)


# In[36]:


from sklearn.metrics import classification_report


# In[37]:


print(classification_report(data['task1'], y_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # NLP Techniques
# # Chemistry publications

# ## In this project:
# ### 1- I will creat a useful dataframe from an SQL database to gain more information on papers that have been published (1996-2016) in  the Journal of the American Chemical Society (JACS).
# ### 2- By  using NLP I will illustrate most frequent titles in two different datasets (JACS and ACS).

# In[1]:


import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud,STOPWORDS
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[2]:


conn = sqlite3.connect("jacs.sqlite")


# In[3]:


cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
cursor.fetchall()


# In[4]:


cursor.execute("PRAGMA table_info(officers)").fetchall()


# In[5]:


authors = pd.read_sql("SELECT * FROM Authors", conn)
authors


# In[6]:


paper_authors = pd.read_sql("SELECT * FROM Paper_Authors", conn)
paper_authors


# In[7]:


papers = pd.read_sql("SELECT * FROM Papers", conn)
papers


# In[8]:


#Joining three tables to prepare a more complete table:
df = pd.read_sql("SELECT * FROM Papers JOIN Paper_Authors ON Papers.paperID=Paper_Authors.paperID JOIN Authors ON Paper_Authors.authorID=Authors.authorID ", conn)
df


# In[55]:


#removing duplicated columns


# In[9]:


df = df.loc[:,~df.columns.duplicated()]
df


# In[10]:


df.isna().sum()


# In[11]:


#Keeping necessary columns
df = df[["DOI", "type", "title", "abstract", "views", "citations", "forename", "surname"]]
df


# In[ ]:


# creating a table from final dataframe
engine = create_engine('sqlite:///jacs.sqlite')
df.to_sql('final', con=engine)


# In[12]:


df.groupby("type").count()


# In[13]:


df1 = df.groupby("type").mean()
df1


# In[14]:


df.corr()


# In[15]:


# Keep in mind, higher views wont necessarily bring higher citations:)


# In[16]:


#Visualizing Views and Citations in each type


# In[17]:


fig, axs = plt.subplots(1, 2, figsize=(13, 4))
sns.barplot(data=df1, x='views', y=df1.index, ax=axs[0], orient='h')
sns.barplot(data=df1, x='citations', y=df1.index, ax=axs[1], orient='h')
plt.title("Views/Citation Mean per type", loc='center')
axs[0].set_title("Mean Views per Type", loc='center')
axs[1].set_title("Mean Citations per Type", loc='center')
plt.tight_layout()
plt.show()


# ## Using the code below you will be able to explore this great dataframe:
# ### By typing just a keyword related to the title you are looking for inside '% %' you will find out more on that title.
# ### You can also change the column and do the same thing on other columns (for example to look up all the papers of a specific author).

# In[18]:


pd.read_sql("SELECT * FROM final WHERE title like '%benzene%' ", conn)


# In[19]:


# Visualizing most common types of publication:


# In[22]:


x = np.char.array(['Addition/Correction',
 'Article',
 'Communication',
 'Others'])
y = np.array([2318,154848,120004,1298])
colors = ["blue", "yellow", "red", "green"]
porcent = 100.*y/y.sum()

explode = [0.1,0,0,0.1]

plt.style.use('fivethirtyeight')

patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2, explode = explode, wedgeprops = {"edgecolor":"black"}, shadow=True)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))
explode = [0,0,0,0.1]
plt.legend(patches, labels, loc='best', bbox_to_anchor=(0.14, 1.),
           fontsize=8)
plt.title('JACS Publications', loc = 'right', fontsize=12)
plt.show()

# plt.savefig('piechart.png', bbox_inches='tight')


# In[23]:


#word tokenization
nltk.download('punkt')
tokenized_messages = df['title'].str.lower().apply(word_tokenize)

print(tokenized_messages)


# In[24]:


# Define a function to returns only alphanumeric tokens
def alpha(text):
    """This function removes all non-alphanumeric characters"""
    alpha = []
    for i in text:
        if i not in ['-','/',',', '(', ')', ':', ';',']', ']', '+', '.', 'by', 'a', 'an', 'and', 'with', 'at', 'as', 'to', 'from', 'in', 'into', 'out', 'of', 'for' 'on', 'under', 'the', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            alpha.append(i)
    return alpha

# Apply our function to tokens
tokenized_messages = tokenized_messages.apply(alpha)
tokenized_messages


# In[25]:


# running the stopwords function on this data was taking so long so I decided to pass the common stopwords into a list and remove them from the text.


# In[26]:


nltk.download('wordnet')
nltk.download('omw-1.4')
def lemmatize(text):
    """This function lemmatize the messages"""
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Create the lemmatized list
    lemmatized = []
    for i in text:
            # Lemmatize and append
            lemmatized.append(lemmatizer.lemmatize(i))
    return " ".join(lemmatized)

# Apply our function to tokens
tokenized_messages = tokenized_messages.apply(lemmatize)
tokenized_messages


# In[27]:


# adding a new column
df['nlp-words'] = tokenized_messages
df


# In[51]:


title = df["nlp-words"].str.cat(sep=', ')

wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color ='white')

# Generate the world clouds
title_wc = wc.generate(title)

# plot the world cloud for spam                     
plt.figure(figsize = (5, 5), facecolor = None) 
plt.imshow(title_wc) 
plt.axis("off") 
plt.title("Common words in titles")
plt.tight_layout(pad = 0) 
plt.show() 
plt.savefig('jacs_wc.png')


# In[29]:


# Well it looks like total synthesis, solid state and carbon nanotube are playing a great role!


# In[30]:


df['nlp-words'] = df['nlp-words'].str.strip()


# In[31]:


# just to see the most common Phrases used in this publications
word_freq = nltk.FreqDist(df["nlp-words"])
rslt=pd.DataFrame(word_freq.most_common(50),columns=['Phrase','Frequency'])
rslt


# ### Now let's run the same codes on our second dataset (ACS):

# In[32]:


data = pd.read_json("chemical_reviews.json")
data


# In[33]:


#It looks so weird! Let's fix it:


# In[34]:


data = data.transpose()
data


# In[35]:


#turning the index to a new column (doi)
data["doi"] = data.index
data = data.reset_index(drop=True)
#drpping unnecessary columns
data.drop(['year', 'journal', 'volume', 'issue', 'page_start', 'page_end'], axis=1, inplace=True)
data


# In[36]:


data.groupby("article_type").count()


# In[37]:


# Visualizing most common types of publication:


# In[50]:


x = np.char.array(['Article',
 'Review',
 'Others'])
y = np.array([4576,2817,333])
colors = ["blue", "red", "green"]
porcent = 100.*y/y.sum()

plt.style.use('fivethirtyeight')

patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2, wedgeprops = {"edgecolor":"black"}, shadow=True)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.title('ACS Publications')    
plt.legend(patches, labels, loc='best', bbox_to_anchor=(0.12, 1.),
           fontsize=11)
plt.show()

# plt.savefig('piechart.png', bbox_inches='tight')


# In[39]:


#word tokenization
nltk.download('punkt')
tokenized_messages = data['title'].str.lower().apply(word_tokenize)

print(tokenized_messages)


# In[40]:


# Define a function to returns only alphanumeric tokens
def alpha(tokens):
    """This function removes all non-alphanumeric characters"""
    alpha = []
    for token in tokens:
        if str.isalpha(token):
            alpha.append(token)
    return alpha

# Apply our function to tokens
tokenized_messages = tokenized_messages.apply(alpha)
print(tokenized_messages)


# In[41]:


# Define a function to remove stop words
def remove_stop_words(tokens):
    """This function removes all stop words in terms of nltk stopwords"""
    no_stop = []
    for token in tokens:
        if token not in stopwords.words('english'):
            no_stop.append(token)
    return no_stop

# Apply our function to tokens
tokenized_messages = tokenized_messages.apply(remove_stop_words)
print(tokenized_messages)


# In[42]:


nltk.download('wordnet')
nltk.download('omw-1.4')
def lemmatize(tokens):
    """This function lemmatize the messages"""
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Create the lemmatized list
    lemmatized = []
    for token in tokens:
            # Lemmatize and append
            lemmatized.append(lemmatizer.lemmatize(token))
    return " ".join(lemmatized)

# Apply our function to tokens
tokenized_messages = tokenized_messages.apply(lemmatize)
print(tokenized_messages)


# In[43]:


data['nlp-words'] = tokenized_messages


# In[44]:


title = data["nlp-words"].str.cat(sep=', ')

wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color ='white', stopwords = set(list(STOPWORDS) + ['chemistry', 'chemical', 'reaction', 'application']))

# Generate the world clouds
title_wc = wc.generate(title)

# plot the world cloud for spam                     
plt.figure(figsize = (5, 5), facecolor = None) 
plt.imshow(title_wc) 
plt.axis("off") 
plt.title("Common words in titles")
plt.tight_layout(pad = 0) 
plt.show() 
plt.savefig('acs1_wc.png')


# In[45]:


# Still synthesis?!


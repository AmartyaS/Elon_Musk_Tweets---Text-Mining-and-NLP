# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 01:46:35 2021

@author: ASUS
"""
# Importing all the required libraries
import re
import nltk
import spacy
import pandas as pd
import gensim
import numpy as np
import seaborn as sns
from PIL import Image
from gensim import corpora
import text2emotion as te
from wordcloud import WordCloud
from textblob import TextBlob
from gensim.models import ldamodel
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the dataset
data=pd.read_csv(r"D:\Data Science Assignments\Python-Assignment\NLP and Text Mining\Elon_musk.csv")
data=pd.DataFrame(data.Text)

# Data Cleaning
text=[]
enr=[]
for i in data.Text:
    raw=re.sub(r'[^a-zA-Z0-9_\s]+',' ',i)
    raw=re.sub(r'\d',' ',raw)
    text.append(raw.lower())
    enr.append(raw)

# Spacy Object
nlp=spacy.load('en_core_web_sm')

# Data Preprocessing
lines=[]
enr_lines=[]
for i in text:
    lines.append(nlp(i))
for i in enr:    
    enr_lines.append(nlp(i))
words=[]
lemmatized=[]
for i in lines:
    words.append([token for token in i if not token.is_stop and token.is_alpha])
    lemmatized.append([token.lemma_ for token in i if not token.is_stop and 
                       len(token) >2 and token.is_alpha])
lis_=[]
for i in lemmatized:
    l=' '.join([token for token in i])
    lis_.append(l)

# Entity Recognition
entity=[]
for i in enr_lines:
    entity.append([(w,w.label_) for w in i.ents ])
entity

# Word Frequency
fdist=FreqDist()
for i in lemmatized:
    for token in i:
            fdist[token]+=1
fdist_max_used_words=pd.DataFrame(fdist.most_common(15),columns=["Words","Count"])
plot=sns.barplot(x=fdist_max_used_words.Words[1:],y=fdist_max_used_words.Count)
plot.set_xticklabels(plot.get_xticklabels(),rotation=45,horizontalalignment='right')

# Topic Modelling
doc_clean=[doc for doc in lemmatized]
dictn=corpora.Dictionary(doc_clean)
dtm=[dictn.doc2bow(doc) for doc in doc_clean]
lda=ldamodel.LdaModel
lda_model=lda(dtm,num_topics=10,id2word=dictn,passes=50)
print(lda_model.print_topics())

# Tfidf 
cv=TfidfVectorizer()
response=cv.fit_transform(lis_)
first_vect=response[0]
vect=pd.DataFrame(first_vect.T.todense(),index=cv.get_feature_names(),columns=["TFIDF_Scores"])
vect.sort_values(by=["TFIDF_Scores"],ascending=False)

# Word2Vec
word2vec=pd.DataFrame(lis_,columns=["Comments"])
word2vec=word2vec.Comments.apply(gensim.utils.simple_preprocess)
model=Word2Vec(word2vec,min_count=1,max_vocab_size=100000)
model.corpus_count
model.epochs
model.wv.key_to_index # Vocabulary list
model.wv.most_similar('space')
model.wv.get_vecattr("spacex","count") # Check word count
len(model.wv) # Length of the vocabulary

# Positive Words
positive_dictn=open(r"D:\Data Science Assignments\Python-Assignment\NLP and Text Mining\positive-words.txt").read()
pos_dictn=nltk.word_tokenize(positive_dictn)
gen_words=' '.join([str(elm) for elm in lis_])
gen_words=nltk.word_tokenize(gen_words)
pos_matches=list(set(pos_dictn).intersection(set(gen_words)))
positive=len(pos_matches)

# Negative Words
negative_dictn=open(r"D:\Data Science Assignments\Python-Assignment\NLP and Text Mining\negative-words.txt").read()
neg_dictn=nltk.word_tokenize(negative_dictn)
neg_matches=list(set(neg_dictn).intersection(set(gen_words)))
negative=len(neg_matches)

# Visualisation of Positive and Negative Words
pos_neg=pd.DataFrame(data=[["Positive_Words",positive],["Negative_Words",negative]],columns=["Polarity","Count"])
sns.barplot(x=pos_neg.Polarity,y=pos_neg.Count)

# Positive Words Wordcloud
pos=' '.join([str(elm) for elm in pos_matches])
mask1=np.array(Image.open("D:\Data_Science Practice\elonmusk.jpg"))
wc=WordCloud(height=512,width=512,mask=mask1,random_state=40,
             max_font_size=14,min_font_size=6,repeat=True,contour_width=0.5,
             contour_color='white',background_color='black',max_words=10000)
wc.generate(pos)
plt.axis('off')
plt.imshow(wc,interpolation='bilinear')
plt.show()

# Negative Words Wordcloud
neg=' '.join([str(elm) for elm in neg_matches])
wc2=WordCloud(height=512,width=512,background_color='black',
              max_font_size=14,min_font_size=5,
              max_words=10000,random_state=40,repeat=True)
wc2.generate(neg)
plt.axis('off')
plt.imshow(wc2,interpolation='bilinear')
plt.show()

# Entire Wordcloud
enwc=' '.join([str(elm) for elm in lis_])
wc2.generate(enwc)
plt.axis('off')
plt.imshow(wc2,interpolation='bilinear')
plt.show()

# Sentiment Analysis
positive_comments=[]
negative_comments=[]
for i in enr:
    polar=TextBlob(i).sentiment.polarity
    if polar>0:
        positive_comments.append(i)
    else:
        negative_comments.append(i)
print("Positive Comments Count : {}".format(len(positive_comments)))
print("Negative Comments Count : {}".format(len(negative_comments)))

# Emotion Mining
emo=te.get_emotion(enwc)
emotion=pd.DataFrame(data=emo.items(),columns=["Emotions","Count"])
emotion.Count=emotion.Count*100
sns.barplot(x=emotion.Emotions,y=emotion.Count)
sns.stripplot(x=emotion.Emotions,y=emotion.Count)
sns.lineplot(data=emotion,x="Emotions",y="Count")
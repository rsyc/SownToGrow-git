# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 07:18:29 2019

@author: user1
"""
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
stop = set(stop_words)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
import re
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.manifold import TSNE
import gensim
import matplotlib.pyplot as plt
import requests
from urllib.request import urlopen
from urllib.parse import urlencode
from json import loads
from mpl_toolkits.mplot3d import Axes3D
import wikipedia



def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #print("stop free: ")
    #print (stop_free)
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #print("punctuation free")
    #print (punc_free)
    num_free= ''.join(ch for ch in punc_free if not ch.isnumeric()) 
    #Abb_free = " ".join([i for i in re.split(r'\s+', punc_free) if i not in stop])
    #print (Abb_free)
    normalized = " ".join(lemma.lemmatize(word) for word in num_free.split())
    return normalized


def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,len(model[word])), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word, topn=10)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    z_coords = Y[:, 2]
    # display scatter plot 2D
    plt.scatter(x_coords, y_coords)
    

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
    '''# display scatter plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(word_labels)): #plot each point + it's index as text above
        ax.scatter(x_coords[i], y_coords[i], z_coords[i], c='r', marker='o')
        ax.text(x_coords[i], y_coords[i], z_coords[i],  '%s' % (word_labels[i]), size=20, zorder=1,  
                color='k') '''

    
    
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
def Abbre_to_complete(subjects):
    Abb_dic={'math':'mathematics', 'pe':'Physical_education',
             'avid':'Advancement_Via_Individual_Determination',
             'coding':'Computer_programming', 'ela':'Language_arts', 
             'irla':'Independent Reading Level Assessment', 'ag':'Agricultural_education',
             'dancepe':'Dance', 'period':'Period_(school)',
             'social study':'Social_studies', 'pal':'Program for Alternative Learning',
             'stem':'Science,_technology,_engineering,_and_mathematics',
             'lote':'Languages_Other_Than_English'}
    if subjects in Abb_dic.keys():
        output=Abb_dic[subjects]
    else:
        output=subjects
    return output


def find_wikitext(name):
    text_list=[]
    #print(name)
    response = requests.get('https://en.wikipedia.org/w/api.php', params={
            'action': 'query',
            'format': 'json',
            'titles': name,
            'prop': 'extracts',
            'exintro': False,
            'explaintext': True,} ).json()
    page = next(iter(response['query']['pages'].values()))
    #page =wikipedia.page(name).content
    if 'extract' in page.keys(): 
        text_list=[name, page['extract']]
    else:
        text_list=[name, '']
        
    
    '''whole_page = wikipedia.page(name).content
    text_list=[name, whole_page]'''
    #params = urlencode({
    #    'format': 'json',
    #    'action': 'parse',
    #    'prop': 'text',
    #    'redirects' : 'true',
    #    'page': name})
    #API = "https://en.wikipedia.org/w/api.php"
    #response = urlopen(API + "?" + params)
    #text_list=[name, response.read().decode('utf-8')]
    return(text_list)
    
    
def getRawPage(page):
    parsed = loads(find_wikitext(page))
    try:
        title = parsed['parse']['title']
        content = parsed['parse']['text']['*']
        return title, content
    except KeyError:
        # The page doesn't exist
        return None, None

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
 
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 03:25:52 2019

@author: user1
"""
import pandas
pandas.set_option('display.max_columns', None)
import functions as fun
from functools import reduce
import time
import numpy as np
# Importing Gensim
import gensim
from gensim.summarization import keywords, textcleaner
from gensim import corpora
from gensim.models import HdpModel
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
from IPython.display import display, HTML
#pyLDAvis.enable_notebook()
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
import nltk


cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
import warnings
import logging # This allows for seeing if the model converges. A log file is created.
#logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




#-------------Start Reading data from csv files--------------------------------
ClassRooms=pandas.read_csv('Classrooms.csv')
ClassRooms.rename(columns={'id':'classroom_id'}, inplace=True)
ClassRooms.rename(columns={'name':'classroom_name'}, inplace=True)
#ClassID=ClassRooms.id
#ClassName=ClassRooms.name

Activities=pandas.read_csv('Activities.csv')
Activities.rename(columns={'id':'activity_id'}, inplace=True)
Activities.rename(columns={'name':'activity_name'}, inplace=True)
#ActID=Activities.id
#ActName=Activities.name

Subjects=pandas.read_csv('Subjects.csv')
Subjects.rename(columns={'name':'subject_name'}, inplace=True)
#ClassID_Sub=Subjects.classroom_id
#SubName=Subjects.name

ClassTimePeriod=pandas.read_csv('ClassroomTimePeriods.csv')
ClassTimePeriod.rename(columns={'id':'classroom_time_period_id'}, inplace=True)

#ClassID_CTP=ClassTimePeriod.classroom_id
#CTP_ID=ClassTimePeriod.id

ClassActMap=pandas.read_csv('ClassroomActivityMapping.csv')
#ActID_map=ClassActMap.activity_id
CTP_ID_map=ClassActMap.classroom_time_period_id
#------------------End of Reading-------------------------------------------
ActID_ClassID_merged=pandas.merge(ClassActMap, ClassTimePeriod, how='left', on=None,\
         left_on='classroom_time_period_id', right_on='classroom_time_period_id',
         left_index=False, right_index=False, sort=False,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
ActID_ClassID_merged=ActID_ClassID_merged.drop(['Unnamed: 0_x', 'id', 'Unnamed: 0_y'], axis=1)

ActNameID_ClassID_merged=pandas.merge(ActID_ClassID_merged, Activities, how='left', on=None,\
         left_on='activity_id', right_on='activity_id',
         left_index=False, right_index=False, sort=False,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
ActNameID_ClassID_merged=ActNameID_ClassID_merged.drop(['Unnamed: 0'], axis=1)

data_frames=[ActNameID_ClassID_merged, Subjects, ClassRooms]
ActClassSub_merged = reduce(lambda  left,right: pandas.merge(left,right,on=['classroom_id'],
                                            how='left'), data_frames)
ActClassSub_merged=ActClassSub_merged.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)

#Remove rows with no subject info (NaN):
ActClassSub_merged = ActClassSub_merged[pandas.notnull(ActClassSub_merged['subject_name'])]
ActClassSub_merged = ActClassSub_merged.fillna('')
#Clean and fix abbreviations in Subject column
cleaned_subs=[fun.clean(name) for name in ActClassSub_merged.subject_name]
cleaned_subs=[fun.clean(name) for name in cleaned_subs]
hist_data=pandas.value_counts(cleaned_subs)
ActClassSub_merged['subject_name']=cleaned_subs
#Clean and fix abbreviations in Activity column
cleaned_acts=[fun.clean(name) for name in ActClassSub_merged.activity_name]
#cleaned_acts=[fun.clean(name) for name in cleaned_acts]
hist_data2=pandas.value_counts(cleaned_acts)
ActClassSub_merged['activity_name']=cleaned_acts
#Clean and fix abbreviations in Activity-description column
cleaned_actDisc=[fun.clean(name) for name in ActClassSub_merged.description_x 
                 if pandas.notnull(name)]
hist_data3=pandas.value_counts(cleaned_actDisc)
if not hist_data3.empty:
    ActClassSub_merged['description_x']=cleaned_actDisc
#Clean and fix abbreviations in classroom_name column
cleaned_className=[fun.clean(name) for name in ActClassSub_merged.classroom_name]
hist_data4=pandas.value_counts(cleaned_actDisc)
if not hist_data4.empty:
    ActClassSub_merged['classroom_name']=cleaned_actDisc
#Clean and fix abbreviations in classroom_description column
cleaned_classDisc=[fun.clean(name) for name in ActClassSub_merged.description_y 
                   if pandas.notnull(name)]
hist_data5=pandas.value_counts(cleaned_classDisc)
if not hist_data5.empty:
    ActClassSub_merged['description_y']=cleaned_classDisc

#========================some tests=====================
math_list=[]
activity_list=[]
for index, rows in ActClassSub_merged.iterrows():
    #print(rows.subject_name)
    if rows.subject_name=='algebra':
        math_list.append([rows.subject_name, rows.activity_name])
        activity_list.append(rows.activity_name)
        
hist_data_mathACT=pandas.value_counts(activity_list)
plt.figure()
plot_data=hist_data_mathACT.where(lambda x : x>=20).dropna() 
plot_data.plot.bar()
plt.show()      
#=============================================================
        
#for subjects in (hist_data.where(lambda x : x>=1000).dropna()).index:
#for rows in ActClassSub_merged['subject_name']=='maths':#subjects:
Row_list =[] 
# Iterate over each row 
for index, rows in ActClassSub_merged.iterrows(): 
    # Create list for the current row
    my_list =[fun.Abbre_to_complete(rows.subject_name), fun.Abbre_to_complete(rows.activity_name)]#, rows.description_x, rows.description_y] 
    # append the list to the final list 
    Row_list.append(my_list)

#Row_list=np.unique(Row_list, axis=0)
    
# Print the list 
#print(Row_list) 
sentences=[]
for subjects in (hist_data.where(lambda x : x>=1000).dropna()).index:
    sentence=[]
    lines=[]
    #subjects=fun.Abbre_to_complete(subjects)
    try:
        wikitext=fun.find_wikitext(subjects) 
        if len(wikitext[1])!=0:
            cleaned_wikitext=fun.clean(wikitext[1])
            lines=nltk.word_tokenize(cleaned_wikitext)#keywords(cleaned_wikitext, deacc=True).split('\n')#gensim.utils.simple_preprocess(wikitext[1])
            '''text_sentences=textcleaner.get_sentences(wikitext[1])
            for jomle in text_sentences:
                #print([jomle])
                lines.append(jomle)'''
                
            #print (subjects)
            for lists in Row_list:
                #print (lists, lists[0])
                if lists[0]==subjects and len(sentence) == 0 and len(lists[1])!=0:
                    sentence=lines+[lists[0]]+lists[1:]
                    #sentence=lines+[fun.Abbre_to_complete(lists[0])]+nltk.word_tokenize(fun.clean(lists[1]))
                elif lists[0]==subjects and len(lists[1])!=0 and len(sentence)!= 0:
                    #sentence+=nltk.word_tokenize(fun.clean(lists[1]))
                    sentence+=lists[1:]
    except ZeroDivisionError:
        for lists in Row_list:
            if lists[0]==subjects and len(sentence) == 0 and len(lists[1])!=0:
                sentence=[lists[0]]+lists[1:]
                #sentence=lines+[fun.Abbre_to_complete(lists[0])]+nltk.word_tokenize(fun.clean(lists[1]))
            elif lists[0]==subjects and len(lists[1])!=0 and len(sentence)!= 0:
                #sentence+=nltk.word_tokenize(fun.clean(lists[1]))
                sentence+=lists[1:]
    sentences.append(sentence)  
# train word2vec on the two sentences
#model.save(subjects+'.model')
#______________________word2vec____________________________________   
#ActClassSub_merged = ActClassSub_merged[pandas.notnull(ActClassSub_merged['subject_name'])]
model = gensim.models.Word2Vec(sentences, size=100, workers=4, window=2, min_count=5, iter=40)
#model = gensim.models.Word2Vec(sentences, size=200, workers=4, window=100, min_count=1, iter=50)


#model.build_vocab(sentences, update = True)  # can be a non-repeatable, 1-pass generator
#model.train(sentences)
#model.wv.most_similar(positive='pe')
fun.tsne_plot(model)
plt.figure()
fun.display_closestwords_tsnescatterplot(model, fun.Abbre_to_complete('math'))
model.similar_by_word(fun.Abbre_to_complete('math'), topn=10)
model.save("standardization"+'.model')

'''
import nltk
import gensim
from nltk.corpus import abc

model= gensim.models.Word2Vec(abc.sents())
X= list(model.wv.vocab)
data=model.most_similar('physics')
print(data)

from gensim.models import Word2Vec
from gensim.test.utils import datapath
gmodel=gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
ms=gmodel.most_similar('science',10)
for x in ms:
    print (x[0],x[1])'''


#for name, val in hist_data.items():  # this is just for test
#    if name =="goal":
#        print (name, val)
'''hist_data=pandas.value_counts(ActClassSub_merged.subject_name)
plt.figure()
ActClassSub_merged.set_index("subject_name",drop=True,inplace=False)
plot_data=hist_data.where(lambda x : x>=1000).dropna() 
plot_data.plot.bar()
plt.show()'''

"""
#Here I combined all the class, activity and subject info into one line to be used for 
#topic modelling:
t = time.time()
docs=[]
for rows in range(0,len(ActClassSub_merged)):
    row_str=''
    for titles in ['subject_name', 'activity_name', 'description_x','classroom_name']:#, 'description_y']:#ActClassSub_merged.columns[3:]:
        if not pandas.isnull(ActClassSub_merged[str(titles)].iloc[rows]):
           row_str+=str(ActClassSub_merged[str(titles)].iloc[rows])+' '
    docs.append(row_str)
print ("Time taken:", time.time()-t)    
   
doc_clean = [fun.clean(doc).split() for doc in docs]  

t = time.time()
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


#_____________________________LDA Model________________________________________
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=45, id2word = dictionary, passes=100)
print ("Time taken:", time.time()-t)    
print(ldamodel.print_topics(num_topics=45, num_words=5))

'''df_topic_sents_keywords = fun.format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix, texts=doc_clean)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

doc_lens = [len(d) for d in df_dominant_topic.Text]'''
#vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary=ldamodel.id2word)
#vis
#pyLDAvis.show(vis, template_type='notebook')

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

lda_topics = ldamodel.show_topics( num_topics=45, num_words=10, log=False, formatted=False)
print ("size of topics:", len(lda_topics))


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lda_train = gensim.models.ldamulticore.LdaMulticore(
                           corpus=doc_term_matrix,
                           num_topics=20,
                           id2word=dictionary,
                           chunksize=100,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)
    lda_train.save('lda_train.model')
#_______________________________________________________________________________
#__________________________________HDP model___________________________________
hdp = HdpModel(doc_term_matrix, dictionary)
topic_info = hdp.print_topics(num_topics=20, num_words=10)
hdp_topics = hdp.show_topics(formatted=False)
#______________________________________________________________________________
#________________________________Plot results__________________________________

topics=lda_topics

fig, axes = plt.subplots(5, 2, figsize=(20,20), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
"""
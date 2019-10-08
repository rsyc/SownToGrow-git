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
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.colors as mcolors
import nltk
from nltk.stem import PorterStemmer    
ps = PorterStemmer() 
from collections import Counter,defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
from itertools import cycle
from sklearn.manifold import TSNE





cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
import logging # This allows for seeing if the model converges. A log file is created.
#logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors


#-------------Start Reading data from csv files--------------------------------
ClassRooms=pandas.read_csv('Classrooms.csv')
ClassRooms.rename(columns={'id':'classroom_id'}, inplace=True)
ClassRooms.rename(columns={'name':'classroom_name'}, inplace=True)
cleaned_class_name=[fun.clean(name) for name in ClassRooms.classroom_name if pandas.notnull(name)]


Activities=pandas.read_csv('Activities.csv')
Activities.rename(columns={'id':'activity_id'}, inplace=True)
Activities.rename(columns={'name':'activity_name'}, inplace=True)
cleaned_act_name=[fun.clean(name) for name in Activities.activity_name if pandas.notnull(name)]


Subjects=pandas.read_csv('Subjects.csv')
Subjects.rename(columns={'name':'subject_name'}, inplace=True)
cleaned_sub_name=[fun.clean(name) for name in Subjects.subject_name if pandas.notnull(name)]

#------------------End of Reading----------------------------------------------

#Merging classRooms and Subjects to be used in Subjects to classes.
data_frames=[ClassRooms, Subjects]
ClassSub_merged =  pandas.merge(left=ClassRooms,right=Subjects, how='left', left_on='classroom_id', right_on='classroom_id')
ClassSub_merged=ClassSub_merged.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'description'], axis=1)


#Finding Unique subject names / classromm names
unique_subs=np.unique(cleaned_sub_name)
unique_classid=np.unique(ClassRooms['classroom_id'])


#CLEANING:
#here we keep original subject name in "keys" and cleaned subject names in "vals"
keys=[]
vals=[]
for names in Subjects['subject_name']:
    if len(names)!=0: #and "_" not in names:
        keys.append(names) 
        vals.append(lemma.lemmatize(fun.Abbre_to_complete(fun.clean(names))))
        
#Make disctionary of of subjects {original name: cleaned name}
subject_dictionary=dict(zip(keys, vals))
list_val=[]
for val in subject_dictionary.values(): 
  if val in list_val: 
    continue 
  else:
    list_val.append(val)
    
list_both=[]
for i in range(len(keys)):
    list_both.append([keys[i],vals[i]])
    

# Separating one-word subjects and multi-word subjects   
new_list_both=[]
long_subname=[]
for i in range(len(list_both)):
    if len(list_both[i][1].split())>1:
        long_subname.append(list_both[i][1])
        new_list_both.append([list_both[i][0], list_both[i][1]])
        
        
#Making bag of words using parts of multi-word subjects and finding the frequency
#of each word in the liast of multi-word subjects    
wordFrequency=fun.word_frequency(long_subname) 
wordFrequency_refined={}  
for key in wordFrequency.keys():
    if wordFrequency[key]>=2:
        wordFrequency_refined[key]=wordFrequency[key]
sorted_wordFrequency=sorted(wordFrequency_refined.items(), key=lambda x: x[1], reverse=True)

# Here we select the most frequent words as the most likely topic for multi-word subjects
# Multi-word subjects are replaced in the original list of subjects with one-word names. 
for i in range(len(list_both)):
    if len(list_both[i][1].split())>1:
        sub_freq=0
        length=0
        sub_name= ''
        for items in list_both[i][1].split():
            item_trans=items#fun.compare_with_RoutinTopics(items)
            if item_trans in wordFrequency_refined.keys():
                if wordFrequency_refined[item_trans]>sub_freq:
                    sub_freq=wordFrequency_refined[item_trans]
                    sub_name= item_trans
                    length=len(item_trans)
                elif wordFrequency[item_trans]<=sub_freq and len(item_trans)>length:
                    sub_freq=wordFrequency_refined[item_trans]
                    sub_name= item_trans
                    length=len(item_trans)
            elif len(sub_name)==0:
                sub_name= sub_name+' '+item_trans
                
        list_both[i][1]=sub_name
    else:
        list_both[i][1]=list_both[i][1]#.replace(" ", "")
        
#-------------------------------------NLP:Vord2Vec-----------------------------
new_name_list=[]
for i in range(len(list_both)):
    new_name_list.append(list_both[i][1])
unique_subs=np.unique(new_name_list)
Counter(new_name_list)
#importing pretranied word2vec model 
glove2word2vec(glove_input_file="glove.6B.100d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
#fun.tsne_plot(glove_model)

#Here we find the vector belong to each subject name using the pretrained word2vec model
#the ones that cannot be found are saved into "notfound" set to be used later.
tokens=[]
labels=[]
notfound=[]
for i in range(len(unique_subs)):
    word=unique_subs[i]
    try:
        tokens.append(glove_model[word])
        labels.append(word)
        #most_similar=glove_model.similar_by_word(ps.stem(list_both[i][1]), topn=10)
        #print(most_similar[0])
        #for j in range(len(most_similar)):
                
    except KeyError:
        word=unique_subs[i].replace(" ", "")
        try:
            tokens.append(glove_model[word])
            labels.append(word)
        except KeyError:
            notfound.append(word)
            #continue
            
#Using subject tokens (vector velues) we find the optimized number of cluster
# that can very well define our data.
centers, clusters = fun.clustering_on_wordvecs(tokens, 10)
centroid_map = dict(zip(labels, clusters))
top_words = fun.get_top_words(labels, 10, centers, np.array(tokens)) 

'''dic_names= defaultdict(list)
dic_values= defaultdict(list)
for word1 in labels:
    for word2 in labels:
        cosine_similarity= glove_model.similarity(word1, word2)
        dic_names[word1].append(word2)
        dic_values[word1].append(cosine_similarity)
        #cosine_similarity = np.dot(glove_model[word1], glove_model[word2])/(np.linalg.norm(glove_model[word1])* np.linalg.norm(glove_model[word2]))
    #max(i for i in dic_values[word1] if i!=1)
    for i in range(len(dic_values[word1])):
        if dic_values[word1][i]>=0.65:
           print(dic_values[word1][i], dic_names[word1][i])'''

# Here for a subject name we find the best cluster it can be assigned to by finding
# the max average similarity value computed by function: "find_wordCluster"
# Then the frequency of the subject/cluster is recorded
Subject_cluster=defaultdict(list)
Subject_cluster_freq={}
for subject_names in new_name_list:
    Selected_cluster=fun.find_wordCluster(glove_model,top_words,subject_names)
    if subject_names not in  Subject_cluster.keys():
        Subject_cluster_freq[subject_names]=1
        Subject_cluster[subject_names].append(Selected_cluster)
    else:
        Subject_cluster_freq[subject_names]+=1


#here we find how the data is distrubuted among different clusters
# subject names not found by pretrained model are all gathers in a group name "OTHERS
clusterFreq={}
notfound_freq={}
AllClusterNames=list(top_words.keys())
AllClusterNames.append('OTHERS')
for cluster in AllClusterNames:
    clusterFreq[cluster]=0
    for subject_names in Subject_cluster.keys():#new_name_list:
        if Subject_cluster[subject_names][0][0]==cluster: #in list(top_words[cluster]):
            clusterFreq[cluster]+=Subject_cluster_freq[subject_names]/len(Subjects)*100#1
sorted(clusterFreq.items(), key=lambda x: x[1], reverse=True)

for subject_names in new_name_list:
    if (subject_names in notfound) and subject_names not in notfound_freq.keys():
        notfound_freq[subject_names]=1
    elif subject_names in notfound:
        notfound_freq[subject_names]+=1       
    
#______________________________________________________________________________           
#____________________________word2vec VISUALIZATION:____________________________________
#______________________________________________________________________________           
#fun.tsne_plot2(tokens,labels)

cmaps = cycle([
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])
for i in range(len(centers)-1):
    col = next(cmaps)
    #plt.figure(i)
    fun.display_cloud(i+1, col,top_words)

embedding_clusters = []
word_clusters = []
for word in labels:
    embeddings = []
    words = []
    for similar_word, _ in glove_model.most_similar(word, topn=10):
        words.append(similar_word)
        embeddings.append(glove_model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)
embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=32)
#embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    
#fun.tsne_plot_similar_words('Similar words', labels, embeddings_en_2d, word_clusters, 0.7,
 #                       'similar_words.png')
#______________________________________________________________________________


#------------------------------ Validation ------------------------------------
# Here List of subject names are compared against dictionary of standard school subjects 
#and list of school abbreviations so abbreviation are replaced with complete form
#and to find a main topic if they are subtopics of a main school topic, e.g. biology
#is included in science subject at schools
        
for i in range(len(list_both)):
    list_both[i][1]=fun.compare_with_RoutinTopics(fun.Satandard_name(list_both[i][1]))#.replace(" ", ""))
    

for i in range(len(list_both)):
    list_both[i][1]=fun.Satandard_name(list_both[i][1].replace(" ", ""))

new_name_list=[]
for i in range(len(list_both)):
    new_name_list.append(list_both[i][1])
len(np.unique(new_name_list))
names=Counter(new_name_list).keys() # equals to list(set(words))
counts=Counter(new_name_list).values() # counts the elements' frequency
sorted_subFrequency=sorted(Counter(new_name_list).items(), key=lambda x: x[1], reverse=True)
subject_length=len(sorted_subFrequency)
sorted_subFrequency2=[]
for i in range(subject_length):
    if sorted_subFrequency[i][1]>1:
        sorted_subFrequency2.append(sorted_subFrequency[i])

#Check if class names would help in grouping similar subjects
df_Sub_both=pandas.DataFrame(list_both)
df_Sub_both.columns=['subject_name','Refined_sub_name']
df_Sub_both.insert(0, 'classroom_id',Subjects['classroom_id'] , True) 
'''ClassSub_dic={}
for i in unique_classid:
    sub_df=ClassSub_merged[ClassSub_merged['classroom_id']==i]
    ref_sub_df=df_Sub_both[df_Sub_both['classroom_id']==i]#.loc[ClassSub_merged2['classroom_id'] == i, 'classroom_name'].iloc[0]
    sub_set=list(ref_sub_df.loc[df_Sub_both['classroom_id'] == i, 'Refined_sub_name'])
    ClassSub_dic[i]=[sub_set]'''
    

#------------------------------------------------------------------------------
'''for inputs in Subjects['subject_name'][0:10]:
    sub_freq=0
    length=0
    sub_name= ''
    print("What you entered: ", inputs)
    inputs=fun.clean(inputs)
    if len(inputs.split())>1:
        sub_freq=0
        length=0
        sub_name= ''
        for items in inputs.split():
            item_trans=fun.compare_with_RoutinTopics(fun.Satandard_name(items.replace(" ", "")))
            if item_trans in names:
                for i in range(len(sorted_subFrequency)):
                    if item_trans==sorted_subFrequency[i][0] and sorted_subFrequency[i][1]>sub_freq:
                        sub_freq=sorted_subFrequency[i][1]
                        sub_name= item_trans
                        length=len(item_trans)
                        original=fun.Satandard_name(items.replace(" ", ""))
                
            elif len(sub_name)==0:
                sub_name= sub_name+' '+item_trans
                original=sub_name
                
            output=sub_name
    else:
        original = fun.Satandard_name(inputs.replace(" ", ""))
        output=fun.compare_with_RoutinTopics(fun.Satandard_name(inputs.replace(" ", "")))
    
    print ("Your standard subject name is: ", original)
    print ("Your subject belong to this general topic: ", output)'''
                
#-----------------------------user interface-----------------------------------
'''user_input=input("Enter subject/activity name: ")
user_input=fun.clean(user_input)
user_input=fun.Abbre_to_complete(user_input)
if len(user_input.split())>1:
    sub_freq=0
    length=0
    sub_name= ''
    for items in user_input.split():
        item_trans=fun.compare_with_RoutinTopics(fun.Satandard_name(items.replace(" ", "")))
        if item_trans in names:
            for i in range(len(sorted_subFrequency)):
                if item_trans==sorted_subFrequency[i][0] and sorted_subFrequency[i][1]>sub_freq:
                    sub_freq=sorted_subFrequency[i][1]
                    sub_name= item_trans
                    length=len(item_trans)
                    original=fun.Satandard_name(items.replace(" ", ""))
                
        elif len(sub_name)==0:
            sub_name= sub_name+' '+item_trans
            original=sub_name
                
        output=sub_name
else:
    original = fun.Satandard_name(user_input.replace(" ", ""))
    output=output=fun.compare_with_RoutinTopics(fun.Satandard_name(user_input.replace(" ", "")))
print ("Your standard subject name is: ", original)
print ("Your subject belong to this general topic: ", output)'''

#--------------------------------Plots-----------------------------------------
plt.figure()
plt.bar(*zip(*clusterFreq.items()))
y_pos = range(len(clusterFreq.items()))
plt.ylabel("Frequency of subject names (%)", fontsize=20)
plt.xticks(y_pos, clusterFreq.keys(), fontsize=20, rotation=90)
plt.tight_layout()


plt.figure()
plt.bar(*zip(*notfound_freq.items()))
y_pos = range(len(notfound_freq.items()))
plt.ylabel("Frequency of not-found subject names (%)", fontsize=18)
plt.xticks(y_pos, notfound_freq.keys(),fontsize=18, rotation=90)
plt.tight_layout()


count=Subjects['subject_name'].value_counts()
df1 = pandas.DataFrame({ 'Topic name':count.index, 'Frequency':count.values})
df1.name = "Raw Topics"
df=df1[df1.iloc[:,1]>= 50]
dff = df[['Topic name','Frequency']]
dff.set_index(["Topic name"],inplace=True)
dff.plot(kind='bar',alpha=1.0)
plt.xlabel("")


df2 = pandas.DataFrame(sorted_subFrequency2)#sorted_subFrequency) #just creating the dataframe
df2.columns=['Topic name','Frequency']
df2.name = "Final Topics"
criteria = df2[ df2.iloc[:,1]>= 50 ] 
criteria.columns=['Topic name','Frequency']
plt.figure()
y_pos = range(len(criteria))#range(len(Counter(new_name_list).keys()))
plt.bar(y_pos,criteria['Frequency'])#(y_pos, Counter(new_name_list).values())
# Rotation of the bars names
plt.xticks(y_pos, criteria['Topic name'], rotation=90)#(y_pos, Counter(new_name_list).keys(), rotation=90)


'''freq1=[]
freq2=[]
shared_topic=[]
for i in range(len(df)):
    if len(df['Topic name'][i].split())==1 and df['Topic name'][i] in list(criteria['Topic name']):
        freq1.append(df['Frequency'][i])
        freq=list(criteria[criteria['Topic name']==df['Topic name'][i]]['Frequency'])
        freq2.append(freq[0])
        shared_topic.append(df['Topic name'][i])
plt.figure()
ax = plt.subplot(111)
y_pos = list(range(len(shared_topic)))
ax.bar(y_pos, [x/len(df1) for x in freq1], width=0.4, color='b', align='center')
ax.bar([x+0.4 for x in y_pos], [x/len(df2) for x in freq2], width=0.4, color='g', align='center')
plt.xticks([x+0.2 for x in y_pos], shared_topic, rotation=90)
plt.ylabel('Frequency of Topics (%)')'''


plt.figure()
y_pos=[1,2]
xlable=['Before Process','After Process']
plt.bar(y_pos,[len(df1), len(df2)])
plt.xticks(y_pos, xlable, rotation=0)
plt.ylabel('Total number of Unique Topics')


unique_count=[]
for i in range(len(sorted_subFrequency2)):
    unique_count.append(sorted_subFrequency2[i][1])
appearance_num=np.zeros((len(np.unique(unique_count)),2), dtype=int)
for i in range(len(np.unique(unique_count))):
    appearance_num[i][0]=np.unique(unique_count)[i]
    for j in range(len(sorted_subFrequency2)):
        if sorted_subFrequency2[j][1]== np.unique(unique_count)[i]:
            appearance_num[i][1]+=1
y_plot=[]
for i in range(len(appearance_num)):
    y_plot.append(appearance_num[i][1])
plt.figure()
y_pos=range(len(np.unique(unique_count)))
xlable=np.unique(unique_count)
plt.bar(y_pos,y_plot)
plt.xticks(y_pos, xlable, rotation=90)
plt.title('')
plt.xlabel('')
plt.ylabel('Topic appearance rate in data')


#------------------------save to excel file------------------------------------
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pandas.ExcelWriter('Topics.xlsx', engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet('Result')
writer.sheets['Result'] = worksheet
worksheet.write_string(0, 0, df1.name)
df1.to_excel(writer,sheet_name='Result',startrow=1 , startcol=0)
worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
df2.to_excel(writer,sheet_name='Result',startrow=1, startcol=df1.shape[1] + 2)

writer.save()


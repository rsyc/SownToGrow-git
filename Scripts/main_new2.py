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
from collections import Counter


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

#ClassID=ClassRooms.id
#ClassName=ClassRooms.name

Activities=pandas.read_csv('Activities.csv')
Activities.rename(columns={'id':'activity_id'}, inplace=True)
Activities.rename(columns={'name':'activity_name'}, inplace=True)
cleaned_act_name=[fun.clean(name) for name in Activities.activity_name if pandas.notnull(name)]

#ActID=Activities.id
#ActName=Activities.name

Subjects=pandas.read_csv('Subjects.csv')
Subjects.rename(columns={'name':'subject_name'}, inplace=True)
cleaned_sub_name=[fun.clean(name) for name in Subjects.subject_name if pandas.notnull(name)]
#cleaned_sub_name=[fun.Abbre_to_complete(name) for name in cleaned_sub_name]
#cleaned_sub_name=[fun.clean(name) for name in cleaned_sub_name]

#Subjects['subject_name']=cleaned_sub_name

'''ClassRooms_copy = ClassRooms
ClassRooms_copy = ClassRooms_copy.set_index('classroom_id')
ClassRooms.columns
Subjects.columns
Subjects_copy = Subjects
Subjects_copy = Subjects_copy.set_index('classroom_id')
Subjects_copy.index

df5 = pandas.concat([ClassRooms_copy, Subjects_copy], axis=1)
print (df5)
df5.to_excel('combined.xlsx')

df4 = pandas.merge(Subjects_copy, ClassRooms_copy, on='classroom_id', how='inner')

df4 = ClassRooms_copy.join(Subjects_copy)
print (df4)
df4.to_excel('combined.xlsx')'''
#ClassID_Sub=Subjects.classroom_id
#SubName=Subjects.name

#------------------End of Reading-------------------------------------------
#data_frames=[ActNameID_ClassID_merged, Subjects, ClassRooms]
data_frames=[ClassRooms, Subjects]
ClassSub_merged =  pandas.merge(left=ClassRooms,right=Subjects, how='left', left_on='classroom_id', right_on='classroom_id')
#reduce(lambda  left,right: pandas.merge(left,right,on=['classroom_id'],
#                                            how='left'), data_frames)
ClassSub_merged=ClassSub_merged.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'description'], axis=1)
#ClassSub_merged.to_excel('combined.xlsx')
unique_subs=np.unique(cleaned_sub_name)
unique_classid=np.unique(ClassRooms['classroom_id'])



keys=[]
vals=[]
for names in Subjects['subject_name']:
    if len(names)!=0: #and "_" not in names:
        keys.append(names) 
        vals.append(fun.clean(names))
        
    
subject_dictionary=dict(zip(keys, vals))
list_val=[]
for val in subject_dictionary.values(): 
  if val in list_val: 
    continue 
  else:
    list_val.append(val)
    
list_both=[]#np.column_stack((keys, vals))
for i in range(len(keys)):
    list_both.append([keys[i],vals[i]])
    
    
new_list_both=[]
long_subname=[]
for i in range(len(list_both)):
    #if list_both[i][0]=='ACC Class':
        #print(list_both[i][0], list_both[i][1], len(list_both[i][1].split()))
    if len(list_both[i][1].split())>1:
        long_subname.append(list_both[i][1])
        new_list_both.append([list_both[i][0], list_both[i][1]])
        #if list_both[i][0]=='ACC Class':
            #print(new_list_both)
        
        
wordFrequency=fun.word_frequency(long_subname) 
wordFrequency_refined={}  
for key in wordFrequency.keys():
    if wordFrequency[key]>=2:
        wordFrequency_refined[key]=wordFrequency[key]
sorted_wordFrequency=sorted(wordFrequency_refined.items(), key=lambda x: x[1], reverse=True)


for i in range(len(list_both)):
    if len(list_both[i][1].split())>1:
        sub_freq=0
        length=0
        sub_name= ''
        for items in list_both[i][1].split():
            item_trans=fun.compare_with_RoutinTopics(items.replace(" ", ""))
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
        list_both[i][1]=list_both[i][1].replace(" ", "")
        
        
#new_name_list=[]
#for i in range(len(list_both)):
#    new_name_list.append(fun.Satandard_name(list_both[i][1]))
#len(np.unique(new_name_list))


#--------------------------------------------------------
for i in range(len(list_both)):
    list_both[i][1]=fun.compare_with_RoutinTopics(list_both[i][1])#.replace(" ", ""))
    

for i in range(len(list_both)):
    list_both[i][1]=fun.Satandard_name(list_both[i][1].replace(" ", ""))
    #print(list_both[i][0])
new_name_list=[]
for i in range(len(list_both)):
    new_name_list.append(list_both[i][1])
len(np.unique(new_name_list))
names=Counter(new_name_list).keys() # equals to list(set(words))
counts=Counter(new_name_list).values() # counts the elements' frequency
sorted_subFrequency=sorted(Counter(new_name_list).items(), key=lambda x: x[1], reverse=True)



df_Sub_both=pandas.DataFrame(list_both)
df_Sub_both.columns=['subject_name','Refined_sub_name']
df_Sub_both.insert(0, 'classroom_id',Subjects['classroom_id'] , True) 
'''ClassSub_dic={}
for i in unique_classid:
    sub_df=ClassSub_merged[ClassSub_merged['classroom_id']==i]
    ref_sub_df=df_Sub_both[df_Sub_both['classroom_id']==i]#.loc[ClassSub_merged2['classroom_id'] == i, 'classroom_name'].iloc[0]
    sub_set=list(ref_sub_df.loc[df_Sub_both['classroom_id'] == i, 'Refined_sub_name'])
    ClassSub_dic[i]=[sub_set]'''
    
'''glove2word2vec(glove_input_file="glove.6B.100d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
for i in range(len(sorted_subFrequency)):
    if sorted_subFrequency[i][1]<=1:
        try:
            most_similar=glove_model.similar_by_word(ps.stem(sorted_subFrequency[i][0]), topn=10)
            print(most_similar[0])
            #for j in range(len(most_similar)):
                
        except KeyError:
            continue'''
#--------------------------------Plots-----------------------------------------
count=Subjects['subject_name'].value_counts()
df1 = pandas.DataFrame({ 'Topic name':count.index, 'Frequency':count.values})
df1.name = "Raw Topics"
df=df1[df1.iloc[:,1]>= 50]
dff = df[['Topic name','Frequency']]
dff.set_index(["Topic name"],inplace=True)
dff.plot(kind='bar',alpha=1.0)
plt.xlabel("")


df2 = pandas.DataFrame(sorted_subFrequency) #just creating the dataframe
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
for i in range(len(sorted_subFrequency)):
    unique_count.append(sorted_subFrequency[i][1])
appearance_num=np.zeros((len(np.unique(unique_count)),2), dtype=int)
for i in range(len(np.unique(unique_count))):
    appearance_num[i][0]=np.unique(unique_count)[i]
    for j in range(len(sorted_subFrequency)):
        if sorted_subFrequency[j][1]== np.unique(unique_count)[i]:
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

# Write each dataframe to a different worksheet.
#df1.to_excel(writer, sheet_name='Sheet1')
#df2.to_excel(writer, sheet_name='Sheet2')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

'''#----------------------------prediction of activity topic-------------------
glove2word2vec(glove_input_file="glove.6B.100d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
for i in range(0,10):
    activity= cleaned_act_name[i]
    topic=fun.find_topic(activity,wordFrequency_refined)
    if len(topic)==0:
        tokens = nltk.word_tokenize(activity)
        for token in tokens: 
            try:
                most_similar=glove_model.similar_by_word(ps.stem(token), topn=10)
                topic=fun.find_topic(most_similar[0],wordFrequency_refined)
            except KeyError:
                
'''
  
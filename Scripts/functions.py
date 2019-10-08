
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say',
                   'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done',
                   'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather',
                   'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run',
                   'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take',
                   'come', 'i','ii','iii','iv','v','vi','th', 'u', 'im', 'th'])
stop = set(stop_words)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
import re
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.manifold import TSNE
import gensim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from urllib.request import urlopen
from urllib.parse import urlencode
from json import loads
from mpl_toolkits.mplot3d import Axes3D
import wikipedia
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter,defaultdict






def clean(doc):
    '''
    INPUT: text
    OUTPUT: cleaned text
    
    stop words, punctuations and one characters are removed. / is replaced by free space
    and the text is normalised. 
    '''
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #print(stop_free)
    word=''
    for ch in stop_free:
        if ch=='/':
            ch=' '
        word+=ch
    
    punc_free = ''.join(ch for ch in word if ch not in exclude)
    #print("punctuation free")
    #print (punc_free)
    num_free= ''.join(ch for ch in punc_free if not ch.isnumeric()) 
    #Abb_free = " ".join([i for i in re.split(r'\s+', punc_free) if i not in stop])
    onechr_free= ' '.join(ch for ch in num_free.split() if len(ch)>1) 
    #normalized = " ".join(lemma.lemmatize(word) for word in num_free.split())
    
    normalized= " ".join(Abbre_to_complete(word) for word in onechr_free.split())
    #if 'pe' in normalized.split():
    #    normalized=normalized.replace("pe", "sport")
    #if normalized=='language art' or normalized=='language arts':
     #   normalized='languageArt'
    return normalized

def process(date):
     date = date.replace("/", " ")
     return(date)

def compare_with_RoutinTopics(word):
    "This function compares input word/subject against a dictionary of rutine school subjects (8 included here)." 

    topic_subtopic_Dic={'history':['history', 'government','world','cultures','geography','human', 'civics'], 
                    'english':['literature','writing','read','reading', 'ela', 'translation', 'english','american','composition','vocabulary','languageart','eld'],
                    'mathematics':['mathematics','algebra','geometry','trigonometry','calculus','prealgebra','analytics','premathmatics','statistics','stat','stats', 'precalculus'],
                    'science':['science','biology','anatomy', 'chemistry', 'physics','physic', 'environmental', 'physical', 'lab','labs'],
                    'language':['language', 'lote', 'chinese', 'french','german','hebrew','italian','japanese','korean','latin','spanish','español','irla','Independent Reading Level Assessment','esol'],
                    'Visual':['visual', 'performing','photography', 'drawing', 'art', 'arts', 'dance', 'drama','theater','theatre', 'music', 'guitar', 'concert','film', 'musical','orchestra','choir','photo','video'],
                    'Physical education':['Physical education','pe','fitness', 'dancepe', 'soccer','waterpolo', 'yoga','athletics','baseball','cheer','tennis','basketball','gym','training','football','lifting'],
                    'economy': [ 'economy', 'finance','microeconomics','macroeconomics']}

    for topics in topic_subtopic_Dic.keys():
        if word.lower().replace(" ", "") in topic_subtopic_Dic[topics]:
            output=topics
            break
        else:
            output=word
    return output

def display_closestwords_tsnescatterplot(model, word):
    "Plots an input word along with the 10 closest word to in using the input pretrained word2vec model."
    
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
    # display scatter plot 2D
    plt.scatter(x_coords, y_coords)
    

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
    

def word_frequency(corpus):
    "counts the number of occurance of a word in a corpus --> to be used in topic modelling"
    wordfreq = {}
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    return wordfreq

def find_topic(activity,wordFrequency_refined):
    tokens = nltk.word_tokenize(activity)
    frequency=0
    for i in range(len(tokens)):
        if tokens[i] in wordFrequency_refined.keys():
            if wordFrequency_refined[tokens[i]]>frequency:
                output=tokens[i]
                frequency=wordFrequency_refined[tokens[i]]
        else:
            output=[]
    return output
                
    
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
    
def tsne_plot2(tokens,labels):
    "Creates and TSNE model and plots it using previously found tokens and labels"
    
    
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
def find_wordCluster(glove_model,top_words,subject_names):
    "Finds the cluster when the input subjectname belongs to by finding the highest value of average similarity computed for each cluster"
    SubDic_Cluster= []#defaultdict(list)
    SubDic_SimValue=[]# defaultdict(list)

    for cluster in top_words.keys():
        cosine_similarity=0
        for names in list(top_words[cluster]):
            try:
                cosine_similarity+= glove_model.similarity(subject_names, names)
            except KeyError:
                continue
        if cosine_similarity!=0:
            average_similarity=cosine_similarity/np.size(top_words,0)
            SubDic_Cluster.append([cluster,average_similarity])
            SubDic_SimValue.append(average_similarity)
    if len(SubDic_SimValue)!=0:
        max_value = max(SubDic_SimValue)
        max_index = SubDic_SimValue.index(max_value)    
        Selected_cluster=SubDic_Cluster[max_index]
    else:
        Selected_cluster=['OTHERS','']
    return Selected_cluster

def clustering_on_wordvecs(word_vectors, num_clusters):
    "This function includes different clustering techniques. At the moment only "
    "Affinity is uncomment. To use other techniques you need to uncomment them."
    
    # Initalize a k-means object and use it to extract centroids
    #kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++', n_init=10, max_iter=1000, tol=1e-03, random_state=0)
    #idx = kmeans_clustering.fit_predict(word_vectors)
    
    #test kmean elbo convergence
    '''distortions = []
    for i in range(1, 10):
        km = KMeans(
                n_clusters=i, init='random',
                n_init=10, max_iter=10000,
                tol=1e-03, random_state=0
                )
        km.fit(word_vectors)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()'''
    
    #hierarchical  clustering
    '''hierarchical_clustering = AgglomerativeClustering().fit(word_vectors)
    AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='ward', memory=None, n_clusters=2,
                        pooling_func='deprecated')
    idx=hierarchical_clustering.predict(word_vectors)
    hierarchical_clustering.labels_'''

    #Affinity clustring    
    Affinity_clustering = AffinityPropagation().fit(word_vectors)
    AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=200, preference=None, verbose=False)
    idx=Affinity_clustering.predict(word_vectors)
    Cluster_centre=Affinity_clustering.cluster_centers_
    
    #DBSCAN clustering
    '''X, labels_true = make_blobs(n_samples=750, centers=Cluster_centre, cluster_std=0.4,
                            random_state=0)

    X = StandardScaler().fit_transform(X)
    DBSCAN_clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    core_samples_mask = np.zeros_like(DBSCAN_clustering.labels_, dtype=bool)
    core_samples_mask[DBSCAN_clustering.core_sample_indices_] = True
    labels=DBSCAN_clustering.labels_
    DBSCAN(algorithm='auto', eps=3, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=2, n_jobs=None, p=None)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    #print('Estimated number of clusters: %d' % n_clusters_)
    #print('Estimated number of noise points: %d' % n_noise_)
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    #print("Adjusted Rand Index: %0.3f"
    #      % metrics.adjusted_rand_score(labels_true, labels))
    #print("Adjusted Mutual Information: %0.3f"
    #      % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, labels))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()'''
    
    
    return Cluster_centre, idx #kmeans_clustering.cluster_centers_, idx

def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs)
    #Closest points for each Cluster center is used to query the closest k points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]
    closest_words_idxs = [x[1] for x in closest_points]
    #Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {}
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i)] = [index2word[j] for j in closest_words_idxs[i][0]]
    #A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words)
    df.index = df.index+1
    return df

def display_cloud(cluster_num, cmap, top_words):
    "Displays clusters of subject names and save them in the running folder"
    wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=cmap)
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num)]]))
                                                                 
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('cluster_' + str(cluster_num), bbox_inches='tight')

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

def Abbre_to_complete(subjects):
    "converts abbreviations existing in the deictionary to their complete format"
    Abb_dic={'calc':'calculus', 'p e':'pe', 'precalc':'precalculus','premath':'premathmatics',
             'p.e.':'pe', 'dance pe':'dancepe', 'social':'social science',
             'acc':'accounting', 'lit':'literature','econ':'economy','eco':'economy'}
             
    if subjects in Abb_dic.keys():
        output=Abb_dic[subjects]
    else:
        output=subjects
    return output

def Abbre_to_complete2(subjects):
    "converts abbreviations existing in the deictionary to their complete format"

    Abb_dic={'calc':'Calculus', 'p e':'pe', 'ap':'Advanced Placement',
             'p.e.':'pe', 'dancepe':'dance pe', 'social':'SocialScience',
             'acc':'accounting','acc':'accounting','bio':'biology','ag':'Agricultural education',
             'coding':'Computer programming', 'chem':'chemistry', 'bio':'biology',
             'biol':'biology', 'irla':'Independent Reading Level Assessment',
             'visual':'Visual and performing arts'}
             
    if subjects in Abb_dic.keys():
        output=Abb_dic[subjects]
    else:
        output=subjects
    return output

def Abbre_to_WikiName(subjects):
    "converts abbreviations existing in the deictionary to their complete format"
    "The complete format is the name that can be addeded to the end of wikipedia page"
    "address so then we can do text scraping."

    Abb_dic={'calc':'Calculus', 'pe':'Physical_education', 'social':'Social_science',
             'accounting':'Accounting', 'math':'Mathematics','Advanced Placement':'Advanced_Placement', 
             'Computer programming':'Computer_programming', 'chemistry':'Chemistry','biology':'Biology',
             'Agricultural education':'Agricultural_education',
             'social study':'Social_studies',
             'stem':'Science,_technology,_engineering,_and_mathematics',
             'lote':'Languages_Other_Than_English','avid':'Advancement_Via_Individual_Determination',
             'period':'Period_(school)', 'Visual and performing arts':'Performing arts'}
             
    if subjects in Abb_dic.keys():
        output=Abb_dic[subjects]
    else:
        output=subjects
    return output

def Satandard_name(subjects):
    "converts abbreviations existing in the deictionary to their standard name format"

    Abb_dic={'math':'mathematics','ap':'Advanced Placement', 'pe':'Physical Education', 
        'coding':'Computer programming', 'chem':'chemistry','bio':'biology', 'biol':'biology',
        'irla':'Independent Reading Level Assessment', 'ag':'Agricultural education',
        'social study':'Social studies', 'pal':'Program for Alternative Learning',
        'stem':'STEM: Science technology engineering mathematics',
        'lote':'Languages Other Than English','avid':'Advancement Via Individual Determination',
        'period':'Period_(school)', 'visual':'Visual and performing arts'}
    #'ela':'Language arts', 
    if subjects in Abb_dic.keys():
        output=Abb_dic[subjects]
    else:
        output=subjects
    return output


def find_wikitext(name):
    "using the name converted by Abbre_to_WikiName, to find the wiki address and scrape first description paragraph"
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
    "finds the title and scrape the whole wiki text"
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
 
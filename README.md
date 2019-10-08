#---Subject Matters:  put your classes into classes---

The code in this repository is in Python.

## Project description
Sown to Grow (www.sowntogrow.com/) has created an online educational platform to help students improve their learning skills. This company wanted to keep it flexible for teachers specially at the beginning of launching the product across country. So teachers had the freedom to enter subject names in any forms they wanted that ended up having unstructured unorganized free form text for subjects. 
This project is about grouping these unstructured subject names into a standardized list of subjects, so then the company would be able to map activities, subjects and classes to this standardized list or groups of subjects. This will be used in future when students' progress has been analysed and best effective activity is needed to be selected and suggested to students for improvement. 

## The data

Data was provided as csv files (5 files: "Classrooms", "Subjects", "Activities", "ClassroomActivityMapping", "ClassroomTimePeriods") by the company. "Subjects" file, which is the main file used in this project included about 7k subject names entered by different teachers across different schools in US. 

## Solution

### Cleaning
Data cleaning (removing stop words, punctuations, numbers, one charecters), 
Making bag of words for multi-word subjects and find the most frequent words,
Assigning new-word (most frequent) to multi-word subjects,
Make a standadized list of subjects (one-word) and their frequency of occurance in the data set,

### Modeling Technique
Pre-trained Word2vec on Wikipedia 2014 + Gigaword 5 was used. 
Vectors (tokens, labels) representing every each subject names (the ones could be recognised by the word dictionary of the word2vec model) were obtained.
Affinity Propagation Clustering was used to find the optimized number of clusters (29 clusters) that describes the data set.

### Results
29 clusters plus one "OTHERS", which includes the subject names not found by the word2vec model, was found.
To show the distribution of data in these clusters, for each subject, an average of Cosine Similarity with each cluster word was found. Subjects were assigned to a cluster with highest average similarity. 

## End product and MVP

Ideal end product will allow to relate user input (including activity, class or subject name) to the list of main clusters found by the model (using the same technique explained in Results section). 

## Presentation Link
https://docs.google.com/presentation/d/1askOMe6Td5dUJDh7yrsgBBNnP77T42v9eqL9rv7t5iY/edit#slide=id.g64a87c635c_0_1

## Recommendations
One good suggestion will be to implement a drop down menue for teachers to pick from a set of sebjects/sub-subjects using the model developed here, to not only decrease variety in naming the same class/subject but also to prevent putting misspelled word and jargons.
It will very well improve the model and its results if the descriptions of subjects/courses were available. 

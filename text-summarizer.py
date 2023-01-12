import nltk                                                 #import libary 
from nltk.corpus import stopwords                           #import libary 
from nltk.cluster.util import cosine_distance               #import libary 
import matplotlib.pyplot as plt                             #import libary 
import matplotlib as mpl                                    #import libary 
import numpy as np                                          #import libary 
import networkx as nx                                       #import libary 

def read_article(file_name):                                #function to read the text file
    file = open(file_name, "r")                             #open the text file as read operation
    filedata = file.readlines()                             #read file one line at a time                                  
    article = filedata[0].split(". ")                       #split each sentences
    sentences = []                                          #create an empty array

    for sentence in article:                                #create a loop based on the number of sentences in the text
        print(sentence)                                     #print all the sentence one line at at time based on the loop
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))     #generate clean sentences
        
    
    return sentences                                        #return the cleaned sentences

def sentence_similarity(sent1, sent2, stopwords=None):      #function for vectorization
    if stopwords is None:                                   #loop to check the stopword
        stopwords = []                                      #create an empty array
 
    sent1 = [w.lower() for w in sent1]                      # separating words
    sent2 = [w.lower() for w in sent2]                      # separating words
 
    all_words = list(set(sent1 + sent2))                    # create a list of unique words
 
    vector1 = [0] * len(all_words)                          # vector 1 array based on the words length
    vector2 = [0] * len(all_words)                          # vector 2 array based on the words length
 
                                                            # build the vector for the first sentence
    for w in sent1:                                         # create a 2d array for vector
        if w in stopwords:                                  # check the stop words
            continue                                        # if there is stopwords, skip it
        vector1[all_words.index(w)] += 1                    # else, vectorize it and insert into array
        #print(vector1)                                                     # print vector 1                      
    #print("------------------------------------------------------------")  # To separate between vector 1 and vector 2 for a better displa
                                                            # build the vector for the second sentence
    for w in sent2:                                         # create a 2d array for vector
        if w in stopwords:                                  #check the stop words
            continue                                        # if there is stopwords, skip it
        vector2[all_words.index(w)] += 1                    # else, vectorize it and insert into array
        #print(vector2)                                      #print vector 2
    #print(1 - cosine_distance(vector1, vector2))            #print sentences similarity
    return 1 - cosine_distance(vector1, vector2)            #return sentences similarity
 
def build_similarity_matrix(sentences, stop_words):         #function to build similarity matrix
                                                            # Create an empty similarity matrix         
    similarity_matrix = np.zeros((len(sentences), len(sentences)))          #return an array filled with zero with the size based on the sentences length
 
    for idx1 in range(len(sentences)):                      # loop to insert value into the array
        for idx2 in range(len(sentences)):                  # loop to insert value into the array
            if idx1 == idx2:                                # check if the sentences are the same
                continue                                    # skip if it is the same
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words) # insert the value to the current position 
    return similarity_matrix                                # return similarity matrix


def generate_summary(file_name,top_n):                    #function to generate summary
    nltk.download("stopwords")                              # download stop words
    stop_words = stopwords.words('english')                 # use english stop words
    summarize_text = []                                     # create an empty array
    
    sentences =  read_article(file_name)                    # split sentences from text

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words) #generate similarity matrix across all the senteces
    print(sentence_similarity_martix)                       #print similarity matrix

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix) # transform the matrix into graph
    
    scores = nx.pagerank(sentence_similarity_graph)         # by using pagerank, calculate the score of each sentence's iteration
    print(scores)                                           #print the scores
    nx.draw(sentence_similarity_graph, with_labels = True)  #produce PageRank graph
    plt.show()                                              #display the graph

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)  #sort the score and pick the sentences with the highest score
    print("Indexes of top ranked_sentence order are ", ranked_sentence)                      #print the sentences according to their rank based on the score

    for i in range(top_n):                                  # loop to create summary based on the number of sentence given for summary
      summarize_text.append(" ".join(ranked_sentence[i][1]))    #create the summary

    print("Summarize Text: \n", ". ".join(summarize_text))  #print the summary

generate_summary( "cloudhub.txt", 2)                          #call the function to begin text summarization, name of the file must be included as parameter for the text to be summarize.

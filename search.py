'''
Created on May 14, 2020

@author: Aaron Wang
'''
# IMPORTANT: PROGRAM ASSUMES THAT DEV FOLDER IS IN SAME DIRECTORY AS PROJECT FILES.

import index
from nltk.stem import PorterStemmer
from posting import Posting
import json
import os
import time
import pickle
import sys
from collections import defaultdict
import math
import heapq
from nltk.corpus import stopwords

if (__name__ == "__main__"):

    cwd = os.getcwd()
    # Load doc_id map from a map.
    mapFile = open(cwd + "\\docidmap.json", 'r')
    mapFilej = json.load(mapFile)

    # Reserved filenames to avoid collisions.
    resFile = ['con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9', 'com0',
               'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9', 'lpt0']
    
    p = PorterStemmer()
    
    # Commmon stop words. Stop words will still be taken in the query, but if they contribute little to the overall result, the
    # calculation will be avoided to minimize query time.
    stopWords = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
                    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                    'being', 'below', 'between', 'both', 'but', 'by', 'cannot', 'could',
                    'did', "didn't", 'do', 'does', 'doing', 'down',
                    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has',
                    'have', 'having', 'he', 'her', 'here',
                    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
                    'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me',
                    'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
                    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
                    'she', 'should', 'so', 'some', 'such',
                    'than', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
                    'these', 'they', 'this', 'those', 'through', 'to',
                    'too', 'under', 'until', 'up', 'very', 'was', 'we',
                    'were', 'what', 'when', 'where', 'which', 'while', 'who',
                    'whom', 'why', 'with', 'would', 'you',
                    'your', 'yours', 'yourself', 'yourselves'])
    stopWords = set([p.stem(i) for i in stopWords])
    
    # Take query
    query = input("Search query: ")
    start = time.perf_counter()
    
    # Separate query into list
    queryList = query.split()
    
    # Break down each element into stem and take out common stop words if the remaining query terms are still reasonable.
    stemList = set([p.stem(i) for i in queryList])
    
    # Stripping stop words.
    for i in stopWords:
        diff = stemList.difference(set([i]))
        if len(diff) != 0:
            stemList = diff
    
    stemFreq = defaultdict(int)
    for i in queryList:
        stemFreq[p.stem(i)] += 1

    # Hard coded path and initialize postingList and postingDict for later use
    path = cwd + "\\results\\"
    postingList =[]
    postingDict = {}
    
    # For each term in stemList, add key = term | value = list of Postings() to postingDict.
    for i in stemList:
        try:
            f = open(cwd + "\\results\\" + i + ".txt", 'rb')
        except FileNotFoundError:
            if (len(i) > 50):
                f = open(cwd + "\\results\\" + i[:50] + "&.txt", 'rb')
            elif i in resFile:
                f = open(cwd + "\\results\\" + i + "&.txt", 'rb')
            else:
                print("Improper query. Make sure query is 0-9a-zA-Z. If it is, term(s) are not in database.")
                sys.exit()
        postingDict[i] = pickle.load(f)
        f.close()

    # Take a lengths and select shortest posting list to start from. Helps with efficiency of intersection operations.
    postingList = sorted([[j.getdocid() for j in i] for i in postingDict.values()], key=lambda l : len(l))
    queryID = set()
    
    # Get set of docids where all query terms are present.
    for i in postingList:
        if (len(queryID) == 0):
            queryID.update(set(i))
        else:
            queryID.intersection_update(set(i))     
            
    # Creates a dictionary with default integer values for each docID.
    # Support for important words is likely to be added here.
    queryResult = {i:0 for i in queryID}
    
    # Iterating through values of postingDict. Cosine similarity is handled here. Important terms are also handled here.
    for i in postingDict.keys():
        for j in postingDict[i]:
            if j.getdocid() in queryID and stemFreq.get(i) != None:
                qtfidf = (1 + math.log(stemFreq[i], 10)) * j.getidf()
                queryResult[j.getdocid()] += (j.gettfidf() * j.getpriority()) * qtfidf
    for i in queryResult.keys():
        queryResult[i] = queryResult[i] / float(mapFilej[str(i)][1])
    counter = 1
    
    # Setting up a max-heap for fast sorting.
    h = [tuple([v,k]) for k,v in queryResult.items()]
    
    heapq._heapify_max(h)
    finish = time.perf_counter()
    
    # Printing query results.
    print("Results:", "(" + str(finish-start) + "s)")
    try:
        for i in range(5):
            m = heapq._heappop_max(h)
            print(i + 1, "-", mapFilej[str(m[1])][0])
    # Excepting IndexError for empty heaps.
    except IndexError:
        pass

    mapFile.close()
    
'''
Created on May 5, 2020

@author: Aaron Wang
'''
# IMPORTANT: PROGRAM ASSUMES THAT DEV FOLDER IS IN SAME DIRECTORY AS PROJECT FILES.

from posting import Posting
from collections import defaultdict
from nltk.stem import PorterStemmer
import nltk
from bs4 import BeautifulSoup
import json
import os
import sys
import re
import math
import concurrent.futures
import shutil
import pickle
import warnings

# Dumps the contents of dictionary d into a .txt file with a name that is decided by dumpCount.
def dump(d, name):
    cwd = os.getcwd()
    if not os.path.exists(cwd + "\\dumps"):
        os.makedirs(cwd + "\\dumps")
    file = open(cwd + "\\dumps\\" + str(name) + ".txt", 'w', encoding='utf-8')
    isFirst = True
    for i in sorted(d.keys()):
        s = i + " "
        for j in sorted(d[i], key=lambda posting : posting.getdocid()):
            if (not isFirst):
                s += ';'
            s += repr(j)
            isFirst = False
        isFirst = True
        s += '\n'
        file.write(s)
    file.close()
    d.clear()

# Similar to dump, but instead dumps docid_map, a dictionary that stores the URL corresponding to a given doc ID.
def dumpMap(docid_map):
    cwd = os.getcwd()
    f = open(cwd + "\\docidmap.json", 'w')
    json.dump(docid_map,f)
    f.close()
    docid_map.clear()

# Dumps result while updating tf-idf.   
def dumpResult(r,docCount):
    df = len(r[1])
    cwd = os.getcwd()
    for i in r[1]:
        i.settfidf((1 + math.log(i.gettfidf(), 10)) * (math.log(docCount/df,10)))
        i.setidf(math.log(docCount/df,10))
    try:
        f = open(cwd + "\\results\\" + r[0] + ".txt", 'wb')
    # Handling reserved file names by adding & to the end if it's a reserved name, and truncating it if it's too long. No colliisons possible as valid terms are 0-9a-zA-Z
    except FileNotFoundError:
        if (len(r[0]) > 50):
            f = open(cwd + "\\results\\" + r[0][:50] + "&.txt", 'wb')
        else:
            f = open(cwd + "\\results\\" + r[0] + "&.txt", 'wb')
    pickle.dump(r[1],f,protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

# Takes a string, from from the dump file, and converts it into a tuple that stores the term in the first index and a list of 
# Postings() in the second. 
def interpret(s):
    st = s.strip()
    spaceIndex = st.find(' ')
    posts = st[spaceIndex+1:].split(';')
    postEval = [eval(s) for s in posts]
    return tuple([st[:spaceIndex], postEval])

# Takes a tuple, in the format described above, and converts it back to a string.
def uninterpret(t):
    st = t[0] + ' '
    first = True
    for i in t[1]:
        if (not first):
            st += ';'
        if type(st) == str:
            st += i
        else:
            st += repr(i)
        first = False
    return st + '\n'

# Updates currentTuple with the next like in each of the files in fileList, if necessary.
def populate(fileList,currentTuple):
    for i in range(len(fileList)):
        if (currentTuple[i] == tuple()):
            s = fileList[i].readline()
            if (s.strip() == ''):
                continue
            else:
                t = interpret(s)
                currentTuple[i] = t

# If currentTuple has multiple indexes that store the same term, merges the Posting lists of these tuples and stores
# it into the lowest index. Old indexes are marked empty with an empty tuple.
def merge(fileList,currentTuple):
    for i in range(len(fileList)):
        try:
            l = []
            for j in currentTuple:
                if (len(j) != 0):
                    l.append(j[0])
                else:
                    l.append('')
            if (currentTuple[i] != tuple()):
                ind = l.index(currentTuple[i][0])
                if (ind != i):
                    combined = currentTuple[i][1] + currentTuple[ind][1]
                    currentTuple[ind] = tuple([currentTuple[i][0], sorted(combined, key=lambda posting : posting.getdocid())])
                    currentTuple[i] = tuple()
        except ValueError:
            pass
        
# Takes currentTuple and returns the index that is alphabetically next in line to be written to the results file.
def firstIndex(currentTuple):
    m = []
    for i in currentTuple:
        if (len(i) != 0):
            m.append(i[0])
        else:
            m.append('')
    l = sorted(m)
    ind = 0
    for i in l:
        if (i != ''):
            break
        ind += 1
    return m.index(l[ind])
    
# Function for multiprocessing. Takes a dirpath, list of filenames, and curr_docid to create dump files -- one for each directory.
def processDirectory(dirpath, filenames, curr_docid):
    
    # PorterStemmer for tokenizing
    p = PorterStemmer()
    
    # index will be the inverted index. It will be offloaded at several points in the program.
    index = defaultdict(list)
    
    # freq is a dictionary that stores the frequency (tf) of each term. Cleared every time a file is finished with parsing.
    freq = defaultdict(int)
    
    # docid map to be returned
    did = {}
    for f in filenames:
        file = open(dirpath + '\\' + f, 'r')
        l = file.readline()
        
        # Tries to load json, with an except statement triggered if there is a Value Error. Should never be triggered.
        try:
            json_dict = json.loads(l)
        except ValueError:
            print('Loading file ' + str(dirpath) + str(f) + ' has failed')
        
        soup = BeautifulSoup(json_dict['content'],features='lxml')
            
        # Updates doc_id map with new URL
        did[curr_docid] = tuple([json_dict['url'], len(soup.get_text())])
        
        
        # Suppress BeautifulSoup warnings about URLs in text.
        warnings.filterwarnings("ignore", category=UserWarning, module = 'bs4')
        
        # Parsing section. Essentially checks to make sure that a stem is greater than 2 characters, but not completely composed of numbers.
        # Temporarily stores the frequency of each word as the tfidf. Updated at the end of indexing.
        for w in nltk.tokenize.word_tokenize(soup.get_text()):
            freq[p.stem(w)] += 1
        for i, j in freq.items():
            if (re.match("^[a-zA-Z0-9][a-zA-Z0-9]+$",i) and (not re.match("^[0-9][0-9]+$", i))):
                index[i].append(Posting(curr_docid, float(j), 1, 0))
        
        # Special Weighing for bold words.
        for w in soup.find_all('b'):
            for t in nltk.tokenize.word_tokenize(w.text):
                if (index.get(t) != None):
                    index[t][-1].setpriority(2)
                    
        # Special weighing for headers.
        for w in soup.find_all(re.compile('^h[1-6]$')):
            for t in nltk.tokenize.word_tokenize(w.text):
                if (index.get(t) != None):
                    index[t][-1].setpriority(3)
                    
        # Special weighing for titles.
        for w in soup.find_all('title'):
            for t in nltk.tokenize.word_tokenize(w.text):
                if (index.get(t) != None):
                    index[t][-1].setpriority(4)
        
        curr_docid += 1
        file.close()
        freq.clear()
    dump(index, os.path.basename(dirpath)) 
    return did

# Cleanup protocol when starting the indexer up again. Removes outdated files.
def cleanup():
    cwd = os.getcwd()
    # Deleting files from past iterations of the program
    print('Cleaning up dumps...', end='')
    if (os.path.isdir(cwd + "\\dumps")):
        shutil.rmtree(cwd + "\\dumps")
    print('done.')
    print('Cleaning up results...', end='')
    if (os.path.isdir(cwd + "\\results")):    
        shutil.rmtree(cwd + "\\results")
    print('done.')
    print('Cleaning up docidmap.json...', end='')
    if (os.path.exists(cwd + "\\docidmap.json")):
        os.remove(cwd + "\\docidmap.json")
    print('done.')

# Confirmation protocol when starting indexer up again in case I'm a dumbass and accidentally run this module instead of search.        
def confirmation():
    inp = input("Starting again will clean up old files. Are you sure you want to start again? [Y/N] : ")
    if (inp.lower() == 'y'):
        cleanup()
    elif (inp.lower() == 'n'):
        sys.exit()
    else:
        confirmation()

if (__name__ == "__main__"):
    # Asks the user if they're sure they want to start indexing. I've ran this program too many times on accident.
    confirmation()
     
    # docid_map is a dictionary storing docids and their corresponding URLs along with the length of the document.
    docid_map = {}
  
    # Keeps track of current doc_id for use in processDirectory
    curr_docid = 0
     
    # Path to the DEV directory, assuming DEV is in the current working directory.
    cwd = os.getcwd()
    path= cwd + "\\DEV"
     
    # Some setting-up for multi-processing.
    executor = concurrent.futures.ProcessPoolExecutor()
    futuresList = []
     
    # Iterates through all directories and uses multiprocessing to process each file in a directory.
    for (dirpath, dirnames, filenames) in os.walk(path):
        futuresList.append(executor.submit(processDirectory, dirpath, filenames, curr_docid))
        curr_docid += len(filenames)
    for f in concurrent.futures.as_completed(futuresList):
        for k,v in f.result().items():
            docid_map[k] = v

    # docCount for use later, namely as N in (1+log(tf) ) * (log(N/df)) when calculating tf-idf.
    docCount = len(docid_map)
    
    # dumpPath and dumpCount (number of dumps) for general use later on.
    dumpPath = cwd + "\\dumps"
    dumpCount = 0
    if (os.path.isdir(dumpPath)):
        dumpCount = len([f for f in os.listdir(dumpPath) if os.path.isfile(os.path.join(dumpPath, f))])
    
    # Transferring docid_map to remote JSON.
    dumpMap(docid_map)
    
    # Making results directory in preparation to store results.
    if not os.path.exists(cwd + "\\results"):
        os.makedirs(cwd + "\\results")
    
    # fileList is a list of opened dump files that are to be used to construct the results files.
    fileList = []
    
    # currentTuple stores the latest tuple from each of the files in fileList.
    currentTuple = [tuple()]*dumpCount
    
    tempCounter = 0
    # For loop for the initial opening of files. Also initial populating of currentTuple.
    for filename in os.listdir(dumpPath):
        fileList.append(open(dumpPath + '\\' + filename, 'r', encoding='utf-8'))
        s = fileList[tempCounter].readline()
        if (s.strip() != ''):
            currentTuple[tempCounter] = interpret(s)
        else:
            currentTuple[tempCounter] = tuple()
        tempCounter += 1 
        
    # Precautionary populate() call to get rid of any empty tuples.
    populate(fileList,currentTuple)
    
    # While loop while currentTuple is not filled with empty tuples. That is, continue to process until there are no lines remaining from all files.
    while (currentTuple != [tuple()]*dumpCount):
        merge(fileList,currentTuple)
        ind = firstIndex(currentTuple)
        dumpResult(currentTuple[ind],docCount)
        currentTuple[ind] = tuple()
        populate(fileList,currentTuple)
     
        
    # Housekeeping - for loop for closing all files in fileList.
    for i in fileList:
        i.close()

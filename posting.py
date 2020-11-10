'''
Created on May 5, 2020

@author: Aaron Wang
'''

class Posting:

# Can implement fields later on.

    def __init__(self, docid, tfidf, priority, idf):
        self.docid = docid
        self.tfidf = tfidf
        self.priority = priority
        self.idf = idf
    
    # Returns string representation of
    def __repr__(self):
        return "Posting(" + str(self.docid) + "," + str(self.tfidf) + "," + str(self.priority) + "," + str(self.idf) + ")"
    
    def getdocid(self):
        return int(self.docid)
    
    def gettfidf(self):
        return float(self.tfidf)
    
    def getpriority(self):
        return int(self.priority)
    
    def getidf(self):
        return float(self.idf)
    
    def settfidf(self, tfidf):
        self.tfidf = tfidf

    def setpriority(self, priority):
        self.priority = priority
        
    def setidf(self, idf):
        self.idf = idf
# Indexer and Basic Search Engine
A Python application for indexing and searching through HTML data from websites.  

## Installation
Move all .py files to any desired directory.  

## Dependencies
BeautifulSoup4, LXML, NLTK

## Usage
Create DEV directory in the same directory as project files.
DEV directory contains directories of domains. Within each domain directory are .json files  
of each web page.

Run index.py to create an index at ./dumps. Each domain will have a corresponding .txt file.  
./results will store serialized files for each search term.  
```bash
> python index.py
Starting again will clean up old files. Are you sure you want to start again? [Y/N] : y
Cleaning up dumps...done.
Cleaning up results...done.
Cleaning up docidmap.json...done.
```

Run search.py to search for domains with a given term.  
Search engine will automatically filter out common terms and return the top 5 results.  
```bash
> python search.py
Search query: cristina lopes
Results: (0.008543300000000364s)
1 - http://sdcl.ics.uci.edu/2016/01/congratulations-lee-martie/
2 - https://www.ics.uci.edu/~lopes/datasets/Koders-log-2007.html
3 - http://mondego.ics.uci.edu/
4 - https://www.ics.uci.edu/~lopes/datasets/sourcerer-maven-aug12.html
5 - https://www.ics.uci.edu/~lopes/datasets/SDS_source-repo-18k.html
```
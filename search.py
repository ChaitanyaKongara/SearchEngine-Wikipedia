import sys, os, re, nltk, Stemmer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english")).union({"reflist", "refbegin", "ref", "citation", "refend", "references", "category", "infobox", "external", "links", "https", "http", "www", "org", "com", "edu", "jpg", "jpeg", "pdf", "png"})
INDEX_DIR = sys.argv[1]
QUERY = sys.argv[2]


def main():
    global QUERY
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    split_query(QUERY)
if __name__ == '__main__':
    main()

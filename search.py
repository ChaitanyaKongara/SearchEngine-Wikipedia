import sys, os, re, nltk, Stemmer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english")).union(
    {
        "reflist",
        "refbegin",
        "ref",
        "citation",
        "refend",
        "references",
        "category",
        "infobox",
        "external",
        "links",
        "https",
        "http",
        "www",
        "org",
        "com",
        "edu",
        "jpg",
        "jpeg",
        "pdf",
        "png",
    }
)
INDEX_DIR = sys.argv[1]
QUERY = sys.argv[2]


def get_value(key, filename):
    low = 0
    high = os.path.getsize(filename) - 1
    with open(filename, "rb") as fd:
        while True:
            mid = (low + high) // 2
            startIdx = mid
            fd.seek(mid)
            while startIdx >= 0 and (fd.read(1) != b"\n" or startIdx == mid):
                fd.seek(-2, 1)
                startIdx -= 1
            startIdx += 1
            if startIdx <= low:
                break
            line = fd.readline().decode().split(":", 1)
            if line[0] == key:
                return line[1]
            elif line[0] > key:
                high = mid - 1
            else:
                low = mid + 1
        line = fd.readline().decode().split(":", 1)
        if line[0] == key:
            return line[1]
        else:
            return ""


def main():
    global QUERY
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    # split_query(QUERY)
    print(get_value("15841678", "title_map"))


if __name__ == "__main__":
    main()

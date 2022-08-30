import sys, os, re, nltk, Stemmer, time
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
K = 10
INDEX_DIR = sys.argv[1]
QUERY_FILE = sys.argv[2]
FIELDS = {"t", "b", "c", "i", "l", "r"}
cachedLines = []
title_weight = 0.1
category_weight = 0.05
match_weight = 0.2

def text_processing(text):
    tokens = re.sub(
        r"\W|_", " ", text.encode("ascii", "ignore").decode()
    )  # remove non alphanumeric
    tokens = [
        token
        for token in tokens.split()
        if token not in STOP_WORDS
        and len(token) > 2
        and not token[0].isnumeric()
        and not token[1].isnumeric()
        and not token.isnumeric()
    ]
    return Stemmer.Stemmer("english").stemWords(tokens)

def split_query(query):
    field_string_mapping = dict()
    idx = 0
    start_idx = 0
    cur_field = "n"
    while idx + 1 < len(query):
        if query[idx] in FIELDS and query[idx + 1] == ":":
            print(query[idx])
            if start_idx != idx:
                if cur_field not in field_string_mapping:
                    field_string_mapping[cur_field] = ""
                field_string_mapping[cur_field] = "".join(
                    [field_string_mapping[cur_field], query[start_idx:idx]]
                )
            cur_field = query[idx]
            idx += 1
            start_idx = idx + 1
        idx += 1
    if start_idx != idx:
        if cur_field not in field_string_mapping:
            field_string_mapping[cur_field] = ""
        field_string_mapping[cur_field] = "".join(
            [field_string_mapping[cur_field], query[start_idx:idx]]
        )
    word_field_freq = dict()
    for field in field_string_mapping:
        for word in text_processing(field_string_mapping[field])
            if word not in word_field_freq:
                word_field_freq[word] = dict()
            if field not in word_field_freq[word]:
                word_field_freq[word][field] = 0
            word_field_freq[word][field] += 1
    return word_field_freq

def get_value(word):
    filename = "/".join([INDEX_DIR, word[0]])
    key = word[1]
    low = 0
    high = os.path.getsize(filename) - 1
    with open(filename, "rb") as fd:
        while True:
            mid = (low + high) // 2
            fd.seek(mid)
            line = fd.readline()
            extraBytes = len(line)
            if mid + extraBytes == high + 1:
                fd.seek(start - end - 1, 1)
                line = fd.readline().decode().split(":", 1)
                if line[0] == key:
                    return (word, line[1])
                else:
                    return None
            

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


def get_Value(key, filename):
    low = 0
    high = os.path.getsize(filename) - 1
    with open(filename, "r") as fd:
        line = fd.readline().split(":", 1)
        while line:
            low += 1
            if line[0] == key:
                return line[1]
            line = fd.readline().split(":", 1)
# doc: {word: tfidf}
def get_doc_tfidf_vectors(word_field_freq):
    tfidf_vectors = dict()
    for word in word_field_freq:
        for doc in get_value(word):
            doc_data = doc.split(" ", 1)
            wc_in_doc = get_value(doc_data[0])
            if doc_data[0] not in tfidf_vectors:
                tfidf_vectors[doc_data[0]] = dict()
            if word not in tfidf_vectors[doc_data]:
                tfidf_vectors[doc_data[0]][word] = 0
            for freq in doc[1].split():
                if freq[0] == "t":
                    tfidf_vectors[doc_data[0]][word] += (title_weight+1+match_weight)*freq[1:] if "t" in word_field_freq[word] else (title_weight+1)*freq[1:]
                elif freq[0] == "c":
                    tfidf_vectors[doc_data[0]][word] += (category_weight+1+match_weight)*freq[1:] if "c" in word_field_freq[word] else (category_weight+1)*freq[1:]
                elif freq[0] in word_field_freq[word]:
                    tfidf_vectors[doc_data[0]][word] += (1+match_weight)*freq[1:]
                else:
                    tfidf_vectors[doc_data[0]][word] += freq[1:]
    return tfidf_vectors


def cosine(v1, v2):
    num = 0
    den1 = 0
    den2 = 0
    for key in v1:
        if key in v2:
            num += v1[key] * v2[key]
        den1 += v1[key] * v1[key]
    for key in v2:
        den2 += v2[key] * v2[key]
    return num/(den1*den2)

def main():
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    # for i in range(26):
    #     cachedLines.append(dict())
    with open(QUERY_FILE) as queryFD:
        queries = queryFD.readlines()
        for query in queries:
            similarity = []
            doc_vectors = get_doc_tfidf_vectors(split_query(query))
            for doc_id in doc_vectors:
                doc_data = get_value(doc_id))
                similarity.append((cosine(doc_vectors[doc_id], query_vector), doc_id, doc_data))
            similarity.sort()
    start = time.time()
    print(get_Value("zyzyk", "index/c"))
    print(time.time() - start)


if __name__ == "__main__":
    main()

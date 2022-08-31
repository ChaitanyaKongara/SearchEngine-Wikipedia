import sys, os, re, nltk, Stemmer, time
from nltk.corpus import stopwords
from sortedcontainers import SortedList

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
title_metadata_file = "title_map"
FIELDS = {"t", "b", "c", "i", "l", "r"}
cachedLines = []
title_weight = 0.1
category_weight = 0.05
match_weight = 0.2
index_wordsize_division = 1


def get_bytesperline():
    bytes_until_line = []
    for i in range(26):
        bytes_until_line.append([0])
        curfile = "/".join([INDEX_DIR, chr(ord("a") + i)])
        with open(curfile, "r") as fd:
            while True:
                lineSize = len(fd.readline())
                if lineSize == 0:
                    break
                bytes_until_line[i].append(bytes_until_line[i][-1] + lineSize)
    bytes_until_line.append([0])
    with open("/".join([INDEX_DIR, title_metadata_file]), "r") as fd:
        while True:
            lineSize = len(fd.readline())
            if lineSize == 0:
                break
            bytes_until_line[26].append(bytes_until_line[26][-1] + lineSize)
    return bytes_until_line


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


# word: {field: cnt}
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
        for word in text_processing(field_string_mapping[field]):
            if word not in word_field_freq:
                word_field_freq[word] = dict()
            if field not in word_field_freq[word]:
                word_field_freq[word][field] = 0
            word_field_freq[word][field] += 1
    return word_field_freq


def calculate_value(prefix):
    curPower = 1
    val = 0
    for char in prefix:
        val += (ord(char) - ord("a")) * curPower
        curPower *= 26
    print(val)
    return val


def get_value(word, title, bytes_until_line):
    filename = INDEX_DIR
    key = word
    filevalue = 26 if title else calculate_value(word[:index_wordsize_division])
    filename = (
        "/".join([INDEX_DIR, title_metadata_file])
        if title
        else "/".join([INDEX_DIR, word[:index_wordsize_division]])
    )
    if not title:
        key = word[index_wordsize_division:]
    low = 0
    high = len(bytes_until_line[filevalue])
    with open(filename, "r") as fd:
        while low < high:
            mid = (low + high) // 2
            fd.seek(bytes_until_line[filevalue][mid])
            line = fd.readline().split(":", 1)
            if line[0] == key:
                return line[1].split(":", 1) if title else line[1]
            if low == mid:
                return None
            if line[0] < key:
                low = mid + 1
            else:
                high = mid


# doc: {word: tfidf}
def get_doc_tfidf_vectors(word_field_freq, bytes_until_line):
    tfidf_vectors = dict()
    for word in word_field_freq:
        for doc in get_value(word, True, bytes_until_line):
            doc_data = doc.split(" ", 1)
            wc_in_doc = get_value(doc_data[0])
            if doc_data[0] not in tfidf_vectors:
                tfidf_vectors[doc_data[0]] = dict()
            if word not in tfidf_vectors[doc_data]:
                tfidf_vectors[doc_data[0]][word] = 0
            for freq in doc[1].split():
                if freq[0] == "t":
                    tfidf_vectors[doc_data[0]][word] += (
                        (title_weight + 1 + match_weight) * freq[1:]
                        if "t" in word_field_freq[word]
                        else (title_weight + 1) * freq[1:]
                    )
                elif freq[0] == "c":
                    tfidf_vectors[doc_data[0]][word] += (
                        (category_weight + 1 + match_weight) * freq[1:]
                        if "c" in word_field_freq[word]
                        else (category_weight + 1) * freq[1:]
                    )
                elif freq[0] in word_field_freq[word]:
                    tfidf_vectors[doc_data[0]][word] += (1 + match_weight) * freq[1:]
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
    return num / (den1 * den2)


def main():
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    bytes_until_line = get_bytesperline()
    similarity = SortedList()
    # for i in range(26):
    #     cachedLines.append(dict())
    # with open(QUERY_FILE) as queryFD:
    #     queries = queryFD.readlines()
    #     for query in queries:
    #         similarity = []
    #         doc_vectors = get_doc_tfidf_vectors(split_query(query), bytes_until_line)
    #         for doc_id in doc_vectors:
    #             doc_data = get_value(doc_id))
    #             if len(similarity) == k: similarity.pop(0)
    #             similarity.add((cosine(doc_vectors[doc_id], query_vector), doc_id, doc_data))
    #         similarity.sort()

    start = time.time()
    print(get_value("cczcwlhyajsc", False, bytes_until_line))
    print(time.time() - start)
    start = time.time()
    print(get_value("17320233", True, bytes_until_line))
    print(time.time() - start)


if __name__ == "__main__":
    main()

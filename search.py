import sys, os, re, nltk, Stemmer, time, math
from nltk.corpus import stopwords
from sortedcontainers import SortedList
from functools import lru_cache

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
query_length = 0
title_metadata_file = "title_map2"
line_metadata_file = "bytesperline"
FIELDS = {"t", "b", "c", "i", "l", "r"}
cachedLines = []
title_weight = 10
category_weight = 5
match_weight = 20
index_wordsize_division = 1
avg_bytes_per_line = 0
total_lines = 0
bytes_until_line = []


def get_bytesperline():
    global avg_bytes_per_line, total_lines, bytes_until_line
    for i in range(26):
        # lines = []
        bytes_until_line.append([0])
        # curfile = "/".join([INDEX_DIR, chr(ord("a") + i)])
        linefile = "/".join(
            [INDEX_DIR, "_".join([chr(ord("a") + i), line_metadata_file])]
        )
        with open(linefile, "r") as fd:
            while True:
                line = fd.readline()
                if len(line) == 0:
                    break
                avg_bytes_per_line += int(line)
                total_lines += 1
                # lines.append("".join([str(lineSize), "\n"]))
                bytes_until_line[i].append(bytes_until_line[i][-1] + int(line))
        # with open(linefile, "w") as linefd:
        #     linefd.writelines(lines)
    bytes_until_line.append([0])
    # with open(
    #     "/".join([INDEX_DIR, "_".join([title_metadata_file, line_metadata_file])]), "r"
    # ) as fd:
    with open("/".join([INDEX_DIR, title_metadata_file]), "r") as fd:
        while True:
            line = fd.readline()
            if len(line) == 0:
                break
            avg_bytes_per_line += int(len(line))
            total_lines += 1
            bytes_until_line[26].append(bytes_until_line[26][-1] + int(len(line)))
    print("Done", avg_bytes_per_line / total_lines)
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
    global query_length
    field_string_mapping = dict()
    idx = 0
    query_length = 0
    start_idx = 0
    cur_field = "n"
    while idx + 1 < len(query):
        if query[idx] in FIELDS and query[idx + 1] == ":":
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
            query_length += 1
    return word_field_freq


def calculate_value(prefix):
    curPower = 1
    val = 0
    for char in prefix:
        val += (ord(char) - ord("a")) * curPower
        curPower *= 26
    return val


# @lru_cache(maxsize=None)
def get_value(word, title):
    # return "10:chaitanya kongara" if title else "639 t1 b2"
    global bytes_until_line
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
            # print(low, mid, high, line[0], key)
            if line[0] == key:
                return line[1][:-1]
            if low == mid:
                return ""
            if line[0] < key:
                low = mid + 1
            else:
                high = mid
        return ""


# doc: {word: tfidf}
def get_tfidf_vectors(word_field_freq):
    global bytes_until_line
    tfidf_vectors = dict()
    query_vector = dict()
    for word in word_field_freq:
        cnt = 0
        for field in word_field_freq[word]:
            cnt += word_field_freq[word][field]
        # can cache
        docs = get_value(word, False)
        if not docs:
            continue
        docs = docs.split(";")
        word_in_docs_cnt = len(docs)
        query_vector[word] = (cnt / query_length) * math.log(
            len(bytes_until_line[26]) / (1 + word_in_docs_cnt)
        )
        # for doc in docs:
        #     doc_data = doc.split(" ", 1)
        #     # print(doc_data[0])
        #     # can cache
        #     wc_in_doc = int(get_value(doc_data[0], True).split(":", 1)[0])
        #     if doc_data[0] not in tfidf_vectors:
        #         tfidf_vectors[doc_data[0]] = dict()
        #     if word not in tfidf_vectors[doc_data[0]]:
        #         tfidf_vectors[doc_data[0]][word] = 0
        #     for freq in doc_data[1].split():
        #         value = 0
        #         if len(freq):
        #             if freq[0] == "t":
        #                 value = int(freq[1:]) * title_weight
        #             elif freq[0] == "c":
        #                 value = int(freq[1:]) * category_weight
        #             if freq[0] in word_field_freq[word]:
        #                 tfidf_vectors[doc_data[0]][word] += value + match_weight * int(
        #                     freq[1:]
        #                 )
        #             else:
        #                 tfidf_vectors[doc_data[0]][word] += value + int(freq[1:])
        #     tfidf_vectors[doc_data[0]][word] = (
        #         min(tfidf_vectors[doc_data[0]][word], wc_in_doc) / wc_in_doc
        #     ) * math.log(len(bytes_until_line[26]) / (1 + word_in_docs_cnt))
    return tfidf_vectors, query_vector


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
    return num / (den1**0.5 * den2**0.5)


def main():
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    global K
    bytes_until_line = get_bytesperline()
    print(bytes_until_line[26][-1])
    similarity = SortedList()
    # for i in range(26):
    #     cachedLines.append(dict())
    with open(QUERY_FILE) as queryFD:
        queries = queryFD.readlines()
        for query in queries:
            start = time.time()
            doc_vectors, query_vector = get_tfidf_vectors(split_query(query))
            print(len(doc_vectors), len(query_vector))
            # for doc_id in doc_vectors:
            #     doc_data = get_value(doc_id, True, bytes_until_line).split(":", 1)
            #     similarity.add(
            #         (cosine(doc_vectors[doc_id], query_vector), doc_id, doc_data[1])
            #     )
            #     if len(similarity) == K + 1:
            #         similarity.pop(0)
            for item in similarity:
                print(item)
            print(time.time() - start)
    # print(get_value.cache_info())


if __name__ == "__main__":
    main()

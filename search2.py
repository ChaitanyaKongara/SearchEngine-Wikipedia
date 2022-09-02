import sys, os, re, nltk, Stemmer, time, math, threading
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
QUERY_OUTFILE = "queries_op.txt"
query_length = 0
title_metadata_file = "title_map2"
line_metadata_file = "bytesperline"
FIELDS = {"t", "b", "c", "i", "l", "r"}
title_weight = 10
category_weight = 5
match_weight = 20
index_wordsize_division = 1
avg_bytes_per_line = 0
total_lines = 0
bytes_until_line = []
word_value = None
title_value = None


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


@lru_cache(maxsize=1000000)
def get_value_word(word):
    global word_value, bytes_until_line
    if word_value:
        return word_value
    filename = INDEX_DIR
    key = word
    filevalue = calculate_value(word[:index_wordsize_division])
    filename = "/".join([INDEX_DIR, word[:index_wordsize_division]])
    key = word[index_wordsize_division:]
    low = 0
    high = len(bytes_until_line[filevalue])
    with open(filename, "r") as fd:
        while low < high:
            mid = (low + high) // 2
            fd.seek(bytes_until_line[filevalue][mid])
            line = fd.readline().split(":", 1)
            word_value = line[1][:-1]
            get_value_word(line[0])
            word_value = None
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


@lru_cache(maxsize=2000000)
def get_value_title(word):
    global bytes_until_line, title_value
    if title_value:
        return title_value
    filename = INDEX_DIR
    key = word
    filevalue = 26
    filename = "/".join([INDEX_DIR, title_metadata_file])
    low = 0
    high = len(bytes_until_line[filevalue])
    with open(filename, "r") as fd:
        while low < high:
            mid = (low + high) // 2
            fd.seek(bytes_until_line[filevalue][mid])
            line = fd.readline().split(":", 1)
            title_value = line[1][:-1]
            get_value_title(line[0])
            title_value = None
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
    for word in sorted(word_field_freq):
        docs = get_value_word(word)
        if not docs:
            continue
        docs = docs.split(";")
        word_in_docs_cnt = len(docs)
        idf = math.log10(len(bytes_until_line[26]) / (word_in_docs_cnt))
        for doc in docs:
            wc = 0
            doc_data = doc.split(" ", 1)
            # print(doc_data[0])
            wc_in_doc = int(get_value_title(doc_data[0]).split(":", 1)[0])
            if doc_data[0] not in tfidf_vectors:
                tfidf_vectors[doc_data[0]] = 0
            for freq in doc_data[1].split():
                value = 0
                if len(freq):
                    if freq[0] == "t":
                        value = int(freq[1:]) * title_weight
                    elif freq[0] == "c":
                        value = int(freq[1:]) * category_weight
                    if freq[0] in word_field_freq[word]:
                        wc += value + match_weight * int(freq[1:])
                    else:
                        wc += value + int(freq[1:])
            tfidf_vectors[doc_data[0]] += min(1, (wc / wc_in_doc)) * idf
    return tfidf_vectors


def main():
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    global K
    bytes_until_line = get_bytesperline()
    print(bytes_until_line[26][-1])
    ofd = open(QUERY_OUTFILE, "w")
    out_lines = []
    with open(QUERY_FILE) as queryFD:
        queries = queryFD.readlines()
        for query in queries:
            similarity = SortedList()
            start = time.time()
            doc_vectors = get_tfidf_vectors(split_query(query.lower()))
            print(len(doc_vectors))
            for doc_id in doc_vectors:
                doc_data = get_value_title(doc_id).split(":", 1)
                similarity.add((doc_vectors[doc_id], doc_id, doc_data))
                if len(similarity) == K + 1:
                    similarity.pop(0)
            for item in similarity:
                out_lines.append(
                    "".join([", ".join([str(item[0]), item[1], item[2][1]]), "\n"])
                )
            out_lines.append("".join([str(time.time() - start), "\n"]))
    ofd.writelines(out_lines)
    ofd.close()
    # avg_time_words = 0
    # avg_time_titles = 0
    # for word in sorted(
    #     [
    #         "afghan",
    #         "andhra",
    #         "america",
    #         "algae",
    #         "ass",
    #         "austwich",
    #         "australia",
    #     ]
    # ):
    #     start = time.time()
    #     get_value_word(word)
    #     avg_time_words += time.time() - start
    #     # print(f"\r{avg_time_words}", end="")
    # for word in sorted(["639", "170052", "10", "2200049", "420", "6969696", "1"]):
    #     start = time.time()
    #     get_value_title(word)
    #     avg_time_titles += time.time() - start
    #     # print(f"\r{avg_time_titles}", end="")
    # print(
    #     avg_time_words,
    #     avg_time_words / 7,
    #     avg_time_titles,
    #     avg_time_titles / 7,
    # )
    print(get_value_word.cache_info())
    print(get_value_title.cache_info())


if __name__ == "__main__":
    main()

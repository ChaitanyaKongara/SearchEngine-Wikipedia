import sys, os, re, Stemmer, time, math, threading
from sortedcontainers import SortedList
from functools import lru_cache
STOP_WORDS = set().union(
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
TOP_A = 10000000000
# TOP_A = 1000
INDEX_DIR = sys.argv[1]
QUERY_FILE = sys.argv[2]
QUERY_OUTFILE = "queries_op.txt"
query_length = 0
title_metadata_file = "title_map"
line_metadata_file = "bytesperline"
FIELDS = {"t", "b", "c", "i", "l", "r"}
title_weight = 1000
category_weight = 500
infobox_weight = 300
rest_weight = 100
match_weight = 20000
index_wordsize_division = 1
avg_bytes_per_line = 0
total_lines = 0
bytes_until_line = []
word_value = None
title_value = None

word_file = INDEX_DIR + "/" + "word_position"
title_file = INDEX_DIR + "/" + "title_position"
word_positions = dict()
title_positions = dict()


def get_bytesperline():
    start = time.time()
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
    with open(
        "/".join([INDEX_DIR, "_".join([title_metadata_file, line_metadata_file])]), "r"
    ) as fd:
        # with open("/".join([INDEX_DIR, title_metadata_file]), "r") as fd:
        while True:
            line = fd.readline()
            if len(line) == 0:
                break
            avg_bytes_per_line += int(line)
            total_lines += 1
            bytes_until_line[26].append(bytes_until_line[26][-1] + int(line))



def prefetch():
    global word_positions, title_positions
    start = time.time()
    with open(word_file, "r") as fd:
        line = fd.readline()
        while len(line):
            line = line.split()
            word_positions[line[0]] = int(line[1])
            line = fd.readline()
    with open(title_file, "r") as fd:
        line = fd.readline()
        dele = set()
        while len(line):
            line = line.split()
            if line[0] in title_positions:
                dele.add(line[0])
                line = fd.readline()
                continue
            title_positions[line[0]] = [int(line[1]), int(line[2])]
            line = fd.readline()
        for item in dele:
            del title_positions[item]
    print("Done with prefetching data", time.time() - start, len(title_positions), len(word_positions))


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
    print(word_field_freq)
    return word_field_freq


def calculate_value(prefix):
    curPower = 1
    val = 0
    for char in prefix:
        val += (ord(char) - ord("a")) * curPower
        curPower *= 26
    return val


# @lru_cache(maxsize=100000)
# def get_value_word(word):
#     global word_value, bytes_until_line, TOP_A
#     if word_value:
#         return word_value
#     filename = INDEX_DIR
#     filevalue = calculate_value(word[:index_wordsize_division])
#     filename = "/".join([INDEX_DIR, word[:index_wordsize_division]])
#     key = word[index_wordsize_division:]
#     low = 0
#     high = len(bytes_until_line[filevalue])
#     with open(filename, "r") as fd:
#         while low < high:
#             mid = (low + high) // 2
#             fd.seek(bytes_until_line[filevalue][mid])
#             line = fd.readline().split(":", 1)
#             cur_value = [pslist.split() for pslist in line[1][:-1].split(";", TOP_A)]
#             if len(cur_value) > TOP_A:
#                 cur_value = cur_value[:-1]
#             word_value = cur_value
#             get_value_word(line[0])
#             word_value = None
#             # print(low, mid, high, line[0], key)
#             if line[0] == key:
#                 return cur_value
#             if low == mid:
#                 return ""
#             if line[0] < key:
#                 low = mid + 1
#             else:
#                 high = mid
#         return ""
#
#
# @lru_cache(maxsize=200000)
# def get_value_title(word):
#     return [43, "abcd ejkfjsldl"]
#     global bytes_until_line, title_value
#     if title_value:
#         return title_value
#     filename = INDEX_DIR
#     key = word
#     filevalue = 26
#     filename = "/".join([INDEX_DIR, title_metadata_file])
#     low = 0
#     high = len(bytes_until_line[filevalue])
#     with open(filename, "r") as fd:
#         while low < high:
#             mid = (low + high) // 2
#             fd.seek(bytes_until_line[filevalue][mid])
#             line = fd.readline().split(":", 2)
#             cur_line = [line[1], line[2][:-1]]
#             title_value = cur_line
#             get_value_title(line[0])
#             title_value = None
#             # print(low, mid, high, line[0], key)
#             if line[0] == key:
#                 return cur_line
#             if low == mid:
#                 return ""
#             if line[0] < key:
#                 low = mid + 1
#             else:
#                 high = mid
#         return ""
#
#

def split(string):
    idx = 0
    text = ""
    data = []
    while idx < len(string):
        if string[idx].isnumeric():
            text = "".join([text, string[idx]])
        else:
            data.append(text)
            text = string[idx]
        idx += 1
    data.append(text)
    return data

@lru_cache(maxsize=10000)
def get_value_word(word):
    global word_positions
    if word not in word_positions:
        return []
    seek_value = word_positions[word]
    key = word[1:]
    file = word[:1]
    with open("/".join([INDEX_DIR, file]), "r") as fd:
        fd.seek(seek_value)
        value = fd.readline()
        # print("--", word, "->", value)
        return [split(pslist) for pslist in value[:-1].split(";")]


@lru_cache(maxsize=200000)
def get_value_title(id):
    global title_positions
    if id not in title_positions:
        return [0, ""]
    seek_value = title_positions[id][0]
    with open("/".join([INDEX_DIR, "title_map"]), "r") as fd:
        fd.seek(seek_value)
        return fd.readline()[:-1].split(":", 1)

# doc: {word: tfidf}
def get_tfidf_vectors(word_field_freq):
    global bytes_until_line, title_positions, word_positions
    tfidf_vectors = dict()
    for word in sorted(word_field_freq):
        docs = get_value_word(word)
        if not docs:
            continue
        word_in_docs_cnt = len(docs)
        idf = math.log10(len(title_positions) / (word_in_docs_cnt))
        for doc_data in docs:
            if not doc_data:
                continue
            wc = 0
            extra = 0
            wc_in_doc = title_positions[doc_data[0]][1]
            if wc_in_doc == 0:
                continue
            if doc_data[0] not in tfidf_vectors:
                tfidf_vectors[doc_data[0]] = 0
            for freq in doc_data[1:]:
                if len(freq):
                    if freq[0] == "t":
                        wc += int(freq[1:]) * title_weight
                        extra += int(freq[1:]) * title_weight
                    elif freq[0] == "c":
                        wc += int(freq[1:]) * category_weight
                        extra += int(freq[1:]) * category_weight
                    elif freq[0] == "i":
                        wc += int(freq[1:]) * infobox_weight
                        extra += int(freq[1:]) * infobox_weight
                    else:
                        wc += int(freq[1:]) * rest_weight
                        extra += int(freq[1:]) * rest_weight
                    if freq[0] in word_field_freq[word]:
                        wc += int(freq[1:]) * match_weight
                        extra += int(freq[1:]) * match_weight
            tfidf_vectors[doc_data[0]] += (wc / (wc_in_doc + extra)) * idf
    return tfidf_vectors


def main():
    if len(sys.argv) < 3:
        sys.exit("ERR: Too few arguments")
    global K
    prefetch()

    ofd = open(QUERY_OUTFILE, "w")
    out_lines = []
    with open(QUERY_FILE) as queryFD:
        queries = queryFD.readlines()
        for query in queries:
            print(query)
            similarity = SortedList()
            start = time.time()
            doc_vectors = get_tfidf_vectors(split_query(query.lower()))
            print("Contender docs count", len(doc_vectors))
            for doc_id in doc_vectors:
                similarity.add((-1 * doc_vectors[doc_id], doc_id))
                if len(similarity) == K + 1:
                    similarity.pop()
            for item in similarity:
                out_lines.append(
                    "".join([", ".join([item[1], get_value_title(item[1])[1]]), "\n"])
                )
            out_lines.append("\n".join([str(time.time() - start), "\n"]))
    ofd.writelines(out_lines)
    ofd.close()
    print(get_value_word.cache_info())
    print(get_value_title.cache_info())


if __name__ == "__main__":
    main()

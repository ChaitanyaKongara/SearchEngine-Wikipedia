import sys, os, shutil, subprocess, bz2, re, xml.sax, Stemmer

STOP_WORDS = set(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    ]
).union(
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
        "files"
    }
)
inline_references = ""
pattern = {
    "i": "\{\{infobox(.+\n)+\}\}",
    "c": "\[\[category:.+\]\]",
    "r": "== ?references ?==\n(.+\n)+\n",
    "l": "== ?external links ?==\n(.+\n)+\n",
    "rs": "== ?references ?==",
    "ls": "== ?external links ?==",
}
page_count = 1
word_count = 0
index_cnt = 0
total_tokens_dump = 0
total_tokens_index = 0
index_word_size_division = 1
temp_dir = "XtempX/"
at_first_id_tag = False
threshold = 100
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sortedcontainers"])
from sortedcontainers import SortedDict, SortedList

inverted_index = SortedDict()
title_file = open("/scratch/saiakarsh/index_files_chaitanya/title_map", "w")
title_id = []
titles_set = set()

def text_processing(text):
    global total_tokens_dump
    tokens = re.sub(
        r"\W|_", " ", text.encode("ascii", "ignore").decode()
    )  # remove non alphanumeric
    temp = tokens.split()
    total_tokens_dump += len(temp)
    tokens = [
        token
        for token in temp
        if token not in STOP_WORDS
        and len(token) > 2
        and not token[0].isnumeric()
        and not token[1].isnumeric()
        and not token.isnumeric()
        and len(token) < 13
        and (len(re.findall("[0-9]", token)) / len(token)) < 0.5 
    ]
    return Stemmer.Stemmer("english").stemWords(tokens)


def extract_field(field, text):
    global pattern, inline_references
    matches = re.finditer(rf"{pattern[field]}", text)
    field_text = inline_references if field == "r" else ""
    for match in matches:
        field_text = " ".join([field_text, match.group()])
    return text_processing(field_text), re.sub(rf"{pattern[field]}", " ", text)


def extract_inline_references(text):
    global inline_references
    references = re.finditer(r"<ref.*>.*</ref>", text)
    inline_references = ""
    for reference in references:
        inline_references = " ".join([inline_references, reference.group()])
    return re.sub(r"<ref.*>.*</ref>", " ", text)


def get_first_index(field, text):
    match_idx = len(text)
    matches = re.finditer(rf"{pattern[field]}", text)
    for match in matches:
        match_idx = match.span(0)[0]
        break
    return match_idx


# refs -> infobox -> Categories -> references -> external links -> body
def extract_fields(text):
    page_field = {}
    text = extract_inline_references(text)
    ref_index = get_first_index("rs", text)
    links_index = get_first_index("ls", text)
    infobox_text = text[: min(ref_index, links_index)]
    rest_text = text[min(ref_index, links_index) :]
    page_field["i"], infobox_text = extract_field("i", infobox_text)
    page_field["b"] = text_processing(re.sub("\{.*?\}", "", infobox_text))
    page_field["c"], rest_text = extract_field("c", rest_text)
    if ref_index != len(text) and links_index != len(text):
        rest_text_ref = rest_text[: abs(ref_index - links_index)]
        rest_text_links = rest_text[abs(ref_index - links_index) :]
        page_field["r"], rest_text_ref = extract_field("r", rest_text_ref)
        page_field["l"], rest_text_links = extract_field("l", rest_text_links)
        page_field["b"] += text_processing(
            re.sub("\{.*?\}", "", rest_text_ref)
        ) + text_processing(re.sub("\{.*?\}", "", rest_text_links))
    else:
        if ref_index != len(text):
            page_field["r"], rest_text = extract_field("r", rest_text)
        elif links_index != len(text):
            page_field["l"], rest_text = extract_field("l", rest_text)
        page_field["b"] += text_processing(re.sub("\{.*?\}", "", rest_text))
    return page_field


def update_inverted_index(page_field):
    global inverted_index, word_count
    for field in page_field:
        if field == "id":
            continue
        word_count += len(page_field[field])
        for word in page_field[field]:
            if not inverted_index.__contains__(word):
                inverted_index[word] = {page_field["id"]: {field: 0}}
            elif page_field["id"] not in inverted_index[word]:
                inverted_index[word][page_field["id"]] = {field: 0}
            if field not in inverted_index[word][page_field["id"]]:
                inverted_index[word][page_field["id"]][field] = 0
            inverted_index[word][page_field["id"]][field] += 1


def write_inverted_index_to_file(index_cnt):
    global inverted_index
    file_name = "".join([temp_dir, str(index_cnt)])
    content = []
    for word, posting_list in inverted_index.items():
        temp_line = ""
        for doc_id, table in posting_list.items():
            if doc_id in titles_set:
                if not temp_line:
                    temp_line = "".join([temp_line, str(doc_id)])
                else:
                    temp_line = ";".join([temp_line, str(doc_id)])
                for field, freq in table.items():
                    if freq:
                        temp_line = " ".join([temp_line, "".join([field, str(freq)])])
        if temp_line:
            content.append("".join([word, "\n", temp_line, "\n"]))
    os.makedirs(temp_dir, exist_ok=True)
    with open(file_name, "w") as file:
        if len(content):
            file.writelines(content)


def close_files(fds):
    for fd in fds:
        fd.close()


def merge_indices(index_cnt):
    print("index_cnt", index_cnt)
    global total_tokens_index
    file_name = temp_dir
    pq = SortedList()
    open_temp_files = []
    cnt = 0
    for idx in range(index_cnt):
        open_temp_files.append(open("".join([file_name, str(idx)]), "r"))
        word = open_temp_files[idx].readline()[:-1]
        if len(word):
            cnt+=1
            pq.add((word, idx))
    print(cnt,"len(pq)")
    os.makedirs(sys.argv[2], exist_ok=True)
    lines_cnt = 0
    index_file_name = pq[0][0][:index_word_size_division]
    index_file = open("".join([sys.argv[2], f"/{index_file_name}"]), "w")
    lines = []
    while len(pq):
        top = pq.pop(0)
        word = top[0]
        if word[:index_word_size_division] != index_file_name:
            index_file.writelines(lines)
            index_file.close()
            lines.clear()
            lines_cnt = 0
            index_file_name = word[:index_word_size_division]
            index_file = open("".join([sys.argv[2], f"/{index_file_name}"]), "w")
        line = ":".join(
            [top[0][index_word_size_division:], open_temp_files[top[1]].readline()[:-1]]
        )
        while len(pq) and pq[0][0] == word:
            top_ = pq.pop(0)
            line = ";".join([line, open_temp_files[top_[1]].readline()[:-1]])
            new_word = open_temp_files[top_[1]].readline()[:-1]
            if len(new_word):
                pq.add((new_word, top_[1]))
        total_tokens_index += 1
        lines_cnt += 1
        lines.append("".join([line, "\n"]))
        if lines_cnt % 100000:
            index_file.writelines(lines)
            lines.clear()
        new_word = open_temp_files[top[1]].readline()[:-1]
        if len(new_word):
            pq.add((new_word, top[1]))
    close_files(open_temp_files)
    # shutil.rmtree(temp_dir)
    if len(lines):
        index_file.writelines(lines)
        lines.clear()
    index_file.close()


class WikiArticleHandler(xml.sax.ContentHandler):
    def startElement(self, tag, attributes):
        global inline_references, at_first_id_tag
        self.tag = tag
        if tag == "page":
            inline_references = ""
            self.id = ""
        elif tag == "title":
            self.title = ""
        elif tag == "text":
            self.text = ""
        elif tag == "id" and not self.id:
            at_first_id_tag = True

    def characters(self, content):
        global at_first_id_tag
        if self.tag == "title":
            self.title = "".join([self.title, content.lower()])
        elif self.tag == "text":
            self.text = "".join([self.text, content.lower()])
        elif self.tag == "id" and at_first_id_tag:
            self.id = "".join([self.id, content.lower()])

    def endElement(self, tag):
        global at_first_id_tag, inverted_index, index_cnt, inline_references, total_tokens_dump, page_count, word_count, pattern
        if tag == "page":
            page_field = extract_fields(self.text)
            page_field["t"] = text_processing(self.title)
            page_field["id"] = self.id
            if word_count > threshold:
                page_count += 1
                update_inverted_index(page_field)
                titles_set.add(self.id)
                title_id.append(
                    ":".join(
                        [
                            str(self.id),
                            str(word_count),
                            "".join(
                                [
                                    re.sub(
                                        r"\\",
                                        " ",
                                        self.title.encode("ascii", "ignore")
                                        .decode()
                                        .strip(),
                                    ),
                                    "\n",
                                ]
                            ),
                        ]
                    )
                )
            word_count = 0
            if page_count % 100000 == 0:
                print("Dealt with article", page_count)
                title_file.writelines(title_id)
                title_id.clear()
                write_inverted_index_to_file(index_cnt)
                index_cnt += 1
                inverted_index.clear()
        elif self.tag == "id":
            at_first_id_tag = False


def main():
    if len(sys.argv) < 5:
        sys.exit("ERR: Too few arguments")
    global temp_dir, index_cnt, inverted_index
    temp_dir = sys.argv[4]
    wiki_dump = bz2.open(sys.argv[1])
    xml.sax.parse(wiki_dump, WikiArticleHandler())
    # xml.sax.parse(open(sys.argv[1], "r"), WikiArticleHandler())
    title_file.writelines(title_id)
    title_id.clear()
    write_inverted_index_to_file(index_cnt)
    index_cnt += 1
    inverted_index.clear()
    print("wc:", word_count)
    print("pc:", page_count)
    merge_indices(index_cnt)
    title_file.close()
    file = open(f"{sys.argv[3]}", "w")
    file.writelines(
        "".join(
            [
                "".join([str(total_tokens_dump), "\n"]),
                "".join([str(total_tokens_index), "\n"]),
            ]
        )
    )
    file.close()


if __name__ == "__main__":
    main()

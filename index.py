import sys, os, shutil, subprocess, bz2, re, xml.sax, nltk, Stemmer
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
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sortedcontainers"])
from sortedcontainers import SortedDict, SortedList

inverted_index = SortedDict()
title_file = open("title_map", "w")
title_id = []


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
    page_field["b"] = text_processing(
        re.sub("\{.*?\}|\[.*?\]|\=\=.*?\=\=", "", infobox_text)
    )
    page_field["c"], rest_text = extract_field("c", rest_text)
    if ref_index != len(text) and links_index != len(text):
        rest_text_ref = rest_text[: abs(ref_index - links_index)]
        rest_text_links = rest_text[abs(ref_index - links_index) :]
        page_field["r"], rest_text_ref = extract_field("r", rest_text_ref)
        page_field["l"], rest_text_links = extract_field("l", rest_text_links)
        page_field["b"] += text_processing(
            re.sub("\{.*?\}|\[.*?\]|\=\=.*?\=\=", "", rest_text_ref)
        ) + text_processing(re.sub("\{.*?\}|\[.*?\]|\=\=.*?\=\=", "", rest_text_links))
    else:
        if ref_index != len(text):
            page_field["r"], rest_text = extract_field("r", rest_text)
        elif links_index != len(text):
            page_field["l"], rest_text = extract_field("l", rest_text)
        page_field["b"] += text_processing(
            re.sub("\{.*?\}|\[.*?\]|\=\=.*?\=\=", "", rest_text)
        )
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
        line = "".join([word, "\n"])
        for doc_id, table in posting_list.items():
            if line[-1] == "\n":
                line = "".join([line, str(doc_id)])
            else:
                line = ";".join([line, str(doc_id)])
            for field, freq in table.items():
                if freq:
                    line = " ".join([line, "".join([field, str(freq)])])
        content.append("".join([line, "\n"]))
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
    for idx in range(index_cnt):
        open_temp_files.append(open("".join([file_name, str(idx)]), "r"))
        word = open_temp_files[idx].readline()[:-1]
        if len(word):
            pq.add((word, idx))
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
        if lines_cnt % 10000:
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
        global inline_references
        self.tag = tag
        if tag == "page":
            inline_references = ""
            self.id = None
        elif tag == "title":
            self.title = ""
        elif tag == "text":
            self.text = ""

    def characters(self, content):
        if self.tag == "title":
            self.title = "".join([self.title, content.lower()])
        elif self.tag == "text":
            self.text = "".join([self.text, content.lower()])
        elif self.tag == "id" and self.id == None:
            self.id = content

    def endElement(self, tag):
        global inverted_index, index_cnt, inline_references, total_tokens_dump, page_count, word_count, pattern
        if tag == "page":
            page_field = extract_fields(self.text)
            page_field["t"] = text_processing(self.title)
            page_field["id"] = self.id
            update_inverted_index(page_field)
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
                                    self.title
                                    .encode("ascii", "ignore")
                                    .decode().strip(),
                                ),
                                "\n",
                            ]
                        ),
                    ]
                )
            )
            word_count = 0
            self.id = None
            if page_count % 10000 == 0:
                print("Dealt with article", page_count)
                title_file.writelines(title_id)
                title_id.clear()
                write_inverted_index_to_file(index_cnt)
                index_cnt += 1
                inverted_index.clear()
            page_count += 1
        if tag == "title":
            total_tokens_dump += len(tuple(re.finditer(r"\s", self.title)))
        if tag == "text":
            total_tokens_dump += len(tuple(re.finditer(r"\s", self.text)))


def main():
    if len(sys.argv) < 4:
        sys.exit("ERR: Too few arguments")
    global index_cnt, inverted_index
    # wiki_dump = bz2.open(sys.argv[1])
    # xml.sax.parse(wiki_dump, WikiArticleHandler())
    xml.sax.parse(open(sys.argv[1], "r"), WikiArticleHandler())
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

import sys, os, shutil, subprocess, bz2, re, xml.sax, nltk, Stemmer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english")).union({"reflist", "refbegin", "ref", "citation", "refend", "references", "category", "infobox", "external", "links", "https", "http", "www", "org", "com", "edu", "jpg", "jpeg", "pdf", "png"})
inline_references = ""
pattern = {
    "i": "\{\{infobox(.+\n)+\}\}",
    "c": "\[\[category:.+\]\]",
    "r": "== ?references ?==\n(.+\n)+\n",
    "l": "== ?external links ?==\n(.+\n)+\n",
    "rs": "== ?references ?==",
    "ls": "== ?external links ?=="
}
page_count = 1
word_count = 1
index_cnt = 0
temp_dir = "XtempX/"
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sortedcontainers"])
from sortedcontainers import SortedDict, SortedList

inverted_index = SortedDict()

def text_processing(text):
    # tokens = re.sub(r"`|~|!|@|#|\$|%|\^|&|\*|\(|\)|-|_|=|\+|\||\\|\[|\]|\{|\}|;|:|'|\"|,|<|>|\.|/|\?|\n|\t", " ", text) # remove non alphanumeric
    tokens = re.sub(r"\W|_", " ", text) # remove non alphanumeric
    tokens = [ token for token in tokens.split() if token not in STOP_WORDS and len(token) > 2 ]
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
    # text = extract_inline_references(text)
    ref_index = get_first_index("rs", text)
    links_index = get_first_index("ls", text)
    infobox_text = text[:min(ref_index, links_index)]
    rest_text = text[min(ref_index, links_index):]
    page_field["i"], infobox_text = extract_field("i", infobox_text)
    page_field["b"] = text_processing(infobox_text)
    page_field["c"], rest_text = extract_field("c", rest_text)
    if ref_index != len(text) and links_index != len(text):
        rest_text_ref = rest_text[:abs(ref_index - links_index)]
        rest_text_links = rest_text[abs(ref_index - links_index):]
        page_field["r"], rest_text_ref = extract_field("r", rest_text_ref)
        page_field["l"], rest_text_links = extract_field("l", rest_text_links)
        page_field["b"] += text_processing(rest_text_ref) + text_processing(rest_text_links)
    else:
        if ref_index != len(text):
            page_field["r"], rest_text = extract_field("r", rest_text)
        elif links_index != len(text):
            page_field["l"], rest_text = extract_field("l", rest_text)       
        page_field["b"] += text_processing(rest_text)
    return page_field

def update_inverted_index(page_field):
    global inverted_index
    for field in page_field:
        if field == "id":
            continue
        for word in page_field[field]:
            if not inverted_index.__contains__(word):
                inverted_index[word] = {
                    page_field["id"]: {
                        field: 0
                    }
                }
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
            if line[-1] == '\n':
                line = "".join([line, str(doc_id)])
            else:
                line = ";".join([line, str(doc_id)])
            for field, freq in table.items():
                if freq:
                    line = " ".join([line, "".join([field, str(freq)])])
        content.append("".join([line,'\n']))
    os.makedirs(temp_dir, exist_ok=True)
    with open(file_name, "w") as file:
        file.writelines(content)

def close_files(fds):
    for fd in fds:
        fd.close()

def merge_indices(index_cnt):
    file_name = temp_dir
    inverted_index_name = "".join([sys.argv[2],"/index"])
    pq = SortedList()
    open_temp_files = []
    for idx in range(index_cnt):
        open_temp_files.append(open("".join([file_name, str(idx)]),"r"))
        word = open_temp_files[idx].readline()[:-1]
        if len(word):
            pq.add((word, idx))
    os.makedirs(sys.argv[2], exist_ok=True)
    with open(inverted_index_name, "w") as index_file:
        lines_cnt = 0
        lines = []
        while len(pq):
            top = pq.pop(0)
            word = top[0]
            line = "\n".join([top[0], open_temp_files[top[1]].readline()[:-1]])
            while len(pq) and pq[0][0] == word:
                top = pq.pop(0)
                line = ";".join([line, open_temp_files[top[1]].readline()[:-1]])
                new_word = open_temp_files[top[1]].readline()[:-1]
                if len(new_word):
                    pq.add((new_word, top[1]))
            lines_cnt += 1
            lines.append("".join([line,"\n"]))
            if lines_cnt % 10000:
                index_file.writelines(lines)
                lines.clear()
            new_word = open_temp_files[top[1]].readline()[:-1]
            if len(new_word):
                pq.add((new_word, top[1]))
    close_files(open_temp_files)
    shutil.rmtree(temp_dir)

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
        global inverted_index, index_cnt, inline_references, page_count, word_count, pattern
        if tag == "page":
            page_field = extract_fields(self.text)
            page_field["t"] = text_processing(self.title)
            page_field["id"] = self.id
            self.id = None
            # for key in page_field:
            #     print(key, ": ", page_field[key])
            #     word_count += len(page_field[key])
            update_inverted_index(page_field)
            if page_count % 50000 == 0:
                write_inverted_index_to_file(index_cnt)
                index_cnt += 1
                inverted_index.clear()
            page_count += 1

def main():
    if len(sys.argv) < 4:
        sys.exit("ERR: Too few arguments")
    global index_cnt
    wiki_dump = bz2.open(sys.argv[1])
    xml.sax.parse(wiki_dump, WikiArticleHandler())
    print("wc:", word_count)
    print("pc:", page_count)
    merge_indices(index_cnt)
if __name__ == '__main__':
    main()

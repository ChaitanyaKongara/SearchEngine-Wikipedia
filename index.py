import sys, bz2, re, xml.sax, nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english")).union({"reflist", "refbegin", "ref", "citation", "refend", "references", "category", "infobox", "external", "links", "https", "http", "www", "org", "com", "edu", "jpg", "jpeg", "pdf", "png"})
STEMMER = SnowballStemmer(language="english")
inline_references = ""

def text_processing(text):
    text = re.sub(r"[\W]", " ", text) # remove non alphanumeric
    tokens = re.sub(r" +", " ", text).strip().split(" ") # remove multiple, trailing, leading whitespaces and split
    tokens = [ STEMMER.stem(token.lower()) for token in tokens if token.lower() not in STOP_WORDS ]
    return tokens

def extract_infobox(text):
    infoboxes = re.finditer(r"\{\{Infobox(.+\n)+\}\}", text)
    infobox_text = ""
    for infobox in infoboxes:
        text = text.replace(infobox.group(), "")
        infobox_text = " ".join([infobox_text, infobox.group()])
    return text_processing(infobox_text), text

def extract_categories(text):
    categories = re.finditer(r"\[\[Category:.+\]\]", text)
    category_text = ""
    for category in categories:
        text = text.replace(category.group(), "")
        category_text = " ".join([category_text, category.group()])
    return text_processing(category_text), text

def extract_inline_references(text):
    references = re.finditer(r"<ref.*>.*</ref>", text)
    references_text = ""
    for reference in references:
        references_text = " ".join([references_text, reference.group()])
    return references_text

def extract_references(text):
    global inline_references
    references = re.finditer(r"== ?References ?==\n(.+\n)+\n", text)
    references_text = inline_references
    for reference in references:
        text = text.replace(reference.group(), "")
        references_text = " ".join([references_text, reference.group()])
    return text_processing(references_text), text

def extract_external_links(text):
    external_links = re.finditer(r"== ?External links ?==\n(.+\n)+\n", text)
    links_text = ""
    for link in external_links:
        text = text.replace(link.group(), "")
        links_text = " ".join([links_text, link.group()])
    return text_processing(links_text), text

# refs -> infobox -> Categories -> External | references -> body
def extract_fields(text):
    page_field = {}
    page_field["i"], text = extract_infobox(text)
    page_field["c"], text = extract_categories(text)
    page_field["r"], text = extract_references(text)
    page_field["l"], text = extract_external_links(text)
    page_field["b"] = text_processing(text)
    return page_field

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
            self.title = "".join([self.title, content])
        elif self.tag == "text":
            self.text = "".join([self.text, content])
        elif self.tag == "id" and self.id == None:
            self.id = content

    def endElement(self, tag):
        global inline_references
        if tag == "page":
            self.id = None
            page_field = extract_fields(self.text)
            page_field["t"] = text_processing(self.title)
            page_field["id"] = self.id
            for key in page_field:
                print(key, ": ", page_field[key])
        elif tag == "text":
            print("self text", self.text)
            inline_references = extract_inline_references(self.text) 

def main():
    wiki_dump = bz2.open(sys.argv[1])
    xml.sax.parse(wiki_dump, WikiArticleHandler())
if __name__ == '__main__':
    main()

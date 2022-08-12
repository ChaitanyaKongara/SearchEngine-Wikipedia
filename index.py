import sys, bz2, re, xml.sax, nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# global variables
STOP_WORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer(language="english")

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
        infobox_text = " ".join([infobox_text, infobox.group().replace("Infobox", "")])
    infobox_text.replace("Infobox","")
    return text_processing(infobox_text), text

def extract_categories(text):
    categories = re.finditer(r"\[\[Category:.+\]\]", text)
    category_text = ""
    for category in categories:
        text = text.replace(category.group(), "")
        category_text = " ".join([category_text, category.group().replace("Category", "")])
    return text_processing(category_text), text

# refs -> infobox -> Categories -> External | references -> body
def extract_fields(text):
    page_field = {}
    page_field["i"], text = extract_infobox(text)
    page_field["c"], text = extract_categories(text)
    
    return page_field

class WikiArticleHandler(xml.sax.ContentHandler):
    def startElement(self, tag, attributes):
        self.tag = tag
        if tag == "page":
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
        if tag == "page":
            self.id = None
        elif tag == "text":
            print("self text", self.text)
            page_field = extract_fields(self.text)
            page_field["t"] = text_processing(self.title)
            page_field["id"] = self.id
            for key in page_field:
                print(key, ": ", page_field[key])

def main():
    wiki_dump = bz2.open(sys.argv[1])
    xml.sax.parse(wiki_dump, WikiArticleHandler())
if __name__ == '__main__':
    main()

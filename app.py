import spacy
import textacy
import os
import pprint
from textacy.extract import keyterms as kt

pp = pprint.PrettyPrinter(indent=4)

def load_text(filename):
    text = ""
    path = os.path.join("./job-descriptions", filename)
    with open(path, 'r') as f:
        text = f.read()
    return text

text = load_text('job1.txt')

en = textacy.load_spacy_lang("en_core_web_lg", disable=("parser",))
doc = textacy.make_spacy_doc(text, lang=en)

pp.pprint(kt.textrank(doc, normalize="lemma", topn=10))
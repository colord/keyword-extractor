from collections import Counter
import os
from pathlib import Path
import spacy
from spacy.matcher import Matcher
import pprint

def load_text(file, existing_dir=False):
    text = None
    path = file if existing_dir else os.path.join('./job-descriptions/', file)
    with open(path, "r") as f:
        text = f.read()
    return text

def add_new_stop_words(load_text, nlp):
    common_words = set(load_text('top1000.txt').split('\n'))
    nlp.Defaults.stop_words |= common_words

def main():
    pp = pprint.PrettyPrinter()

    nlp = spacy.load("en_core_web_lg")

    add_new_stop_words(load_text, nlp)

    matcher = Matcher(nlp.vocab)
    # + is for one or more instances of a proper noun
    pattern = [
        {"POS": "PROPN", "IS_STOP": False, "OP": "*"}, 
        # {"POS": "NOUN", "IS_STOP": False, "OP": "*"},
        {"POS": "ADJ", "IS_STOP": False, "OP": "*"},
    ]
    matcher.add("PROPER_NOUNS", [pattern], greedy="LONGEST")

    results = {}
    for file in Path('./job-descriptions').glob('*.txt'):
        if file != 'top1000.txt':
            text = load_text(file, existing_dir=True)
            doc = nlp(text)

            matches = matcher(doc)

            for match_id, start, end in matches:
                key = doc[start:end].text
                results[key] = results.get(key, 0) + 1
                
    pp.pprint(results)

if __name__ == "__main__":
    main()
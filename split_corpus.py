"""
Splits the VUAMC corpus into its respective genres
Expects the full British National Corpus XML edition to be available at data/2554/download
"""

from pathlib import Path
from collections import defaultdict

from bs4 import BeautifulSoup

with open("data/VUAMC.xml", "r") as f:
    document = f.read()
xml = BeautifulSoup(document, features="lxml")

genres = {}
basepath = Path("data/2554/download/Texts")
corpuses = defaultdict(list)
for text in xml.find_all("text"):
    if "xml:id" in text.attrs:
        id = text["xml:id"]
        frag = id[:3].upper()
        if frag not in genres:
            filepath = basepath / frag[:1] / frag[:2] / frag[:3]
            filepath = filepath.with_suffix(".xml")
            with open(filepath, "r", encoding="utf8") as f:
                text = f.read()
            text = BeautifulSoup(text, features="lxml")
            genre = text.find("classcode").text

            # determine genre within the 4 VUAMC classes
            genre = genre.split()
            type, genre, *_ = genre
            if genre.startswith("ac"):
                genre = "ac"

            genres[frag] = genre
        genre = genres[frag]
        corpuses[genre].append(id)

for genre, ids in corpuses.items():
    with open(f"data/{genre}.xml", "w", encoding="utf8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:vici="http://www.tei-c.org/ns/VICI">\n')
        f.write('<text>\n<group>\n')
        for id in ids:
            start = document.find(f'<text xmlns="http://www.tei-c.org/ns/1.0" xml:id="{id}">')
            end = document.find('</text>', start) + len('</text>')
            text = document[start:end]
            f.write("    " + text + "\n")
        f.write('</group>\n</text>\n</TEI>\n')
    print(f"Written {genre}.xml")

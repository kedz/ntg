import os
import sys
import argparse
from urllib.parse import quote
import urllib.request
import json
import time


query_template = "https://{lang}.wikipedia.org/w/api.php?" \
   "action=query&list=categorymembers&cmtitle={title}" \
   "&format=json&cmlimit=500&cmtype={type}"

def get_pages(cat, lang):
    query = query_template.format(
        lang=lang, type="page", title=quote(cat.replace(" ", "_")))
    result = urllib.request.urlopen(query)
    data = json.load(result)
    category = data['query']['categorymembers']
    
    pages = []
    for x in category:
        title = x["title"]
        if title.startswith("Template:"):
            continue
        if title.startswith("User:"):
            continue
        if title.startswith("List of "):
            continue
        pages.append(title)
    return pages


def get_subcats(cat, lang):
    query = query_template.format(
        lang=lang, type="subcat", title=quote(cat.replace(" ", "_")))
    result = urllib.request.urlopen(query)
    data = json.load(result)
    category = data['query']['categorymembers']
    
    cats = []
    for x in category:
        title = x["title"]
        if title.startswith("Template:"):
            continue
        if title.startswith("User:"):
            continue
        if title.startswith("List of "):
            continue
        cats.append(title)
    return cats


def get_cols():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns)


def clear_screen():
    sys.stdout.write("\r" + " " * get_cols())
    sys.stdout.write("\r")
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Download page titles underneath a wikipedia category.")
    parser.add_argument("--category", required=True, type=str)
    parser.add_argument(
        "--lang", required=False, type=str, choices=["simple", "en"],
        default="simple")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir != "" and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    queue = [args.category]

    found_categories = set()
    found_pages = set()

    completed = 0
    with open(args.output, "w") as f:
        while len(queue) > 0:
            current_cat = queue.pop(0)
            clear_screen()
            msg = "{} completed | {} remain | {} pages | current {}".format(
                completed, len(queue), len(found_pages), 
                current_cat.split(":")[1])
            sys.stdout.write(msg[:get_cols()])
            sys.stdout.flush()

            for page in get_pages(current_cat, args.lang):
                if page not in found_pages:
                    found_pages.add(page)
                    f.write(page)
                    f.write("\n")
                    f.flush()

            for cat in get_subcats(current_cat, args.lang):
                if cat not in found_categories:
                    found_categories.add(cat)
                    queue.append(cat)

            completed += 1
    print("")
    

#urllib.error.HTTPError:         


import os
import sys
import argparse
from collections import defaultdict
import wikipedia
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re

def get_cols():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns)
COL = get_cols()

class NoHTMLException(Exception):
    def __init__(self):
        super(Exception, self).__init__("No html found!")

def read_names(path):
    names = set()
    with open(path, "r") as f:
        for line in f:
            names.add(line.strip())
    return sorted(list(names))


def get_page(name, prefix):
    wikipedia.set_lang(prefix)
    page = wikipedia.page(name)
    title = str(page.title)
    url = str(page.url)
    content = page.content.split("\n\n")[0].strip()
    content = re.sub(r"[\s\n\r\t]+", r" ", content)
    html = get_html_intro(page.html())
    if html is None: 
        raise NoHTMLException()
    html = re.sub(r"[\s\n\r\t]+", r" ", html)

    return title, url, content, html


def get_simple_wp_page(name):
    return get_page(name, "simple")


def get_wp_page(name, swp_content):
    try:
        return get_page(name, "en")
    except wikipedia.exceptions.DisambiguationError as e:
        swp_tokens = set(word_tokenize(swp_content.lower()))
        
        best_match = None
        best_score = 0
        
        for page in e.options:
            da_title, da_url, da_content, da_html = get_page(page, "en")
            da_tokens = set(word_tokenize(da_content.lower()))
            score = len(swp_tokens.intersection(da_tokens))
            if score > best_score:
                best_score = score
                best_match = (da_title, da_url, da_content, da_html)

        if best_match:
            return best_match
        else:
            raise e


def get_html_intro(html):

    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all("p"):
        if len(element.getText().strip()) > 3:
            return str(element).strip()




def clear_screen():
    sys.stdout.write("\r" + " " * COL)
    sys.stdout.write("\r")
    sys.stdout.flush()


TEMPLATE = "{swp_cats}\t{swp_title}\t{swp_url}\t{swp_content}\t{swp_html}\t" \
    "{wp_title}\t{wp_url}\t{wp_content}\t{wp_html}\n"


def cache_previous_try(path):
    cache = {}
    with open(path, "r") as f:
        f.readline()
        for line in f:
            items = line.strip().split("\t")
            assert(len(items) == 9)
            cache[items[5]] = line
    return cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download parallel simple/regular wp corpus.")
    parser.add_argument(
        "--category-paths", nargs="+", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        cache = cache_previous_try(args.output)
    else:
        cache = {}

    output_dir = os.path.dirname(args.output)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_names = defaultdict(list)
    for path in args.category_paths:
        category = os.path.splitext(os.path.basename(path))[0].lower()
        names = read_names(path)
        for name in names:
            all_names[name].append(category)

    names_cats = sorted(all_names.items(), key=lambda x: x[0])

    with open(args.output + ".tmp", "w") as f:
        f.write("swp_cats\tswp_title\tswp_url\tswp_content\tswp_html\t")
        f.write("wp_title\twp_url\twp_content\twp_html\n")
        for i, (name, cats) in enumerate(names_cats, 1):
            
            cat_str = "|".join(cats)

            clear_screen()
            msg = "{}/{} page {} ({})".format(
                i, len(names_cats), name, cat_str)
            sys.stdout.write(msg[:COL])
            sys.stdout.flush()

            if name in cache:
                f.write(cache[name])
                del cache[name]
                f.flush()
                continue

            try:
                swp_title, swp_url, swp_text, swp_html = get_simple_wp_page(
                    name)
                wp_title, wp_url, wp_text, wp_html = get_wp_page(
                    name, swp_text)
                line = TEMPLATE.format(
                    swp_cats=cat_str, swp_title=swp_title, swp_url=swp_url, 
                    swp_content=swp_text, swp_html=swp_html,
                    wp_title=wp_title, wp_url=wp_url, wp_content=wp_text, 
                    wp_html=wp_html)

                f.write(line)
                f.flush()
            except wikipedia.exceptions.DisambiguationError as e:
                print("\ndisambig: {}".format(name))
                continue
            except wikipedia.exceptions.PageError as e:
                print("\npage error: {}".format(name))
                continue
            except NoHTMLException as e:
                print("\nno html: {}".format(name))
                continue
            except wikipedia.exceptions.WikipediaException as e:
                print("\nunknown: {}".format(name))
                continue
        
    print("")

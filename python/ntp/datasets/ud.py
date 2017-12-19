import os
from .util import get_data_dir, download_url_to_file


def get_conllu_path(lang="en", subset="all", split="train"):
    
    if subset == "all":
        subsets = available_subsets(lang)
    else:
        subsets = [subset]

    paths = []
    for subset in subsets:
        path = os.path.join(
            get_data_dir(), "ud", lang, subset, 
            "ud.{}.{}.{}.conllu".format(lang, subset, split))
        if not os.path.exists(path):
            download_ud(lang, subset, split, path)
        paths.append(path)

    if len(paths) == 1:
        paths = paths[0]

    return paths

def available_languages():
    return ["en", "es", "fr"]

def available_subsets(lang):
    code2subsets = {
        "en": ["original", "lines"],
        "fr": ["original"],
        "es": ["ancora", "original"]}
    return code2subsets[lang]

def ud_urls(lang, subset, split):
    m = {
        "en": {
            "original": {
                "train": "https://github.com/UniversalDependencies/UD_English/raw/master/en-ud-train.conllu",
                "valid": "https://github.com/UniversalDependencies/UD_English/raw/master/en-ud-dev.conllu",
                "test": "https://github.com/UniversalDependencies/UD_English/raw/master/en-ud-test.conllu"},
            "lines": {
                "train": "https://github.com/UniversalDependencies/UD_English-LinES/raw/master/en_lines-ud-train.conllu",
                "valid": "https://github.com/UniversalDependencies/UD_English-LinES/raw/master/en_lines-ud-dev.conllu",
                "test": "https://github.com/UniversalDependencies/UD_English-LinES/raw/master/en_lines-ud-test.conllu"},
        },
        "es": {
            "ancora": {
                "train": "https://github.com/UniversalDependencies/UD_Spanish-AnCora/raw/master/es_ancora-ud-train.conllu",
                "valid": "https://github.com/UniversalDependencies/UD_Spanish-AnCora/raw/master/es_ancora-ud-dev.conllu",
                "test": "https://github.com/UniversalDependencies/UD_Spanish-AnCora/raw/master/es_ancora-ud-test.conllu"
            },
            "original": {
                "train": "https://github.com/UniversalDependencies/UD_Spanish/raw/master/es-ud-train.conllu",
                "valid": "https://github.com/UniversalDependencies/UD_Spanish/raw/master/es-ud-dev.conllu",
                "test": "https://github.com/UniversalDependencies/UD_Spanish/raw/master/es-ud-test.conllu"
            }
        },
        "fr": {
            "original": {
                "train": "https://github.com/UniversalDependencies/UD_French/raw/master/fr-ud-train.conllu",
                "valid": "https://github.com/UniversalDependencies/UD_French/raw/master/fr-ud-dev.conllu",
                "test": "https://github.com/UniversalDependencies/UD_French/raw/master/fr-ud-test.conllu"
                }
        }
    }
    return m[lang][subset][split] 



def download_ud(lang, subset, split, path):
    url = ud_urls(lang, subset, split)
    print(url)
    print(path)
    print("Downloading\n  {}\nto\n  {} ...".format(url, path))
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    download_url_to_file(url, path)
    if not os.path.exists(path):
        raise Exception("Failed to download url {} to {}".format(
            url, path))

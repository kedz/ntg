import os
import tarfile
import re
import nltk

from .util import get_data_dir


def get_imdb_data_path(split="train"):

    if split not in ["train", "test"]:
        raise Exception("split must be either 'train' or 'test'.")

    file_path = os.path.join(
        get_data_dir(), "imdb", "imdb.{}.tsv".format(split))  

    if not os.path.exists(file_path):
        download_imdb_data()

    return file_path

def download_imdb_data():
    root_path = "/home/kedz/Downloads/aclImdb"

    tar_path = "/home/kedz/Downloads/aclImdb_v1.tar.gz"

    train_path = os.path.join(get_data_dir(), "imdb", "imdb.train.tsv") 
    test_path = os.path.join(get_data_dir(), "imdb", "imdb.test.tsv") 
    
    with tarfile.open(tar_path) as tar_fp:
        with open(train_path, "w") as tr_fp, open(test_path, "w") as te_fp:

            tr_fp.write("label\trating\ttext\n")
            te_fp.write("label\trating\ttext\n")

            for member in tar_fp.getmembers():
                items = member.name.split("/")
                
                if len(items) != 4:
                    continue
                
                _, part, label, filename = items

                if part not in ["train", "test"]:
                    continue

                if label not in ["pos", "neg"]:
                    continue

                review_id, rating = os.path.splitext(filename)[0].split("_")
                member_fp = tar_fp.extractfile(member)
                text = member_fp.read()
                text = text.decode("utf8")
                text = re.sub(r"<br.*?>", r" ", text)
                tokens = nltk.word_tokenize(text)
     
                line = "{}\t{}\t{}\n".format(
                    label, rating, " ".join(tokens))

                if part == "train":
                    tr_fp.write(line)
                else:
                    te_fp.write(line)

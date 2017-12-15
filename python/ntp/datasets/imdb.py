import ntp
import torch
import os
import urllib.request
import tarfile
import re
import nltk
import io
import sys


from .util import get_data_dir, download_url_to_buffer


def get_imdb_data_path(split="train"):

    if split not in ["train", "test", "unsup"]:
        raise Exception("split must be either 'train', 'test', or 'unsup'.")

    file_path = os.path.join(
        get_data_dir(), "imdb", "imdb.{}.tsv".format(split))  

    if not os.path.exists(file_path):
        download_imdb_data()

    return file_path

def download_imdb_data():

    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    print("Downloading imdb data from {} ...".format(url))
    fileobj = download_url_to_buffer(url)
    
    train_path = os.path.join(get_data_dir(), "imdb", "imdb.train.tsv") 
    test_path = os.path.join(get_data_dir(), "imdb", "imdb.test.tsv") 
    unsup_path = os.path.join(get_data_dir(), "imdb", "imdb.unsup.tsv") 

    if not os.path.exists(os.path.dirname(train_path)):
        os.makedirs(os.path.dirname(train_path))
    
    total_train = 0
    total_test = 0
    total_unsup = 0

    status_tmp = "\rpreprocessed {:0.3f}% train | {:0.3f}% test " \
        "| {:0.3f}% unsup"

    with tarfile.open(fileobj=fileobj) as tar_fp:
        with open(train_path, "w") as tr_fp, open(test_path, "w") as te_fp, \
                open(unsup_path, "w") as unsup_fp:

            tr_fp.write("label\trating\ttext\n")
            te_fp.write("label\trating\ttext\n")
            unsup_fp.write("text\n")

            for member in tar_fp.getmembers():
                items = member.name.split("/")
                
                if len(items) != 4:
                    continue
                
                _, part, label, filename = items

                if part not in ["train", "test"]:
                    continue

                if label not in ["pos", "neg", "unsup"]:
                    continue

                if part == "train" and label != "unsup":
                    total_train += 1
                elif part == "test":
                    total_test += 1
                else:
                    total_unsup += 1

                sys.stdout.write(
                    status_tmp.format(
                        total_train / 25000 * 100,
                        total_test / 25000 * 100,
                        total_unsup / 50000 * 100))
                sys.stdout.flush()

                review_id, rating = os.path.splitext(filename)[0].split("_")
                member_fp = tar_fp.extractfile(member)
                text = member_fp.read()
                text = text.decode("utf8")
                text = re.sub(r"<br.*?>", r" ", text)
                tokens = nltk.word_tokenize(text)
     
                if part == "train" and label in ["neg", "pos"]:
                    line = "{}\t{}\t{}\n".format(
                        label, rating, " ".join(tokens))
                    tr_fp.write(line)
                elif part == "test":
                    line = "{}\t{}\t{}\n".format(
                        label, rating, " ".join(tokens))
                    te_fp.write(line)
                else:
                    unsup_fp.write("{}\n".format(" ".join(tokens)))
    print("") 

def get_imdb_dataset(split="train", at_least=5, start_token="__START__", 
                     stop_token=None, lower=True, replace_digit="#"):

    file_path_template = os.path.join(
        get_data_dir(), "imdb", 
        "imdb.part={part}.at_least={at_least}.start_token={start_token}." \
        "stop_token={stop_token}.lower={lower}." \
        "replace_digit={replace_digit}.bin")

    file_path = file_path_template.format(
        part=split,
        at_least=at_least,
        start_token=start_token if start_token else "",
        stop_token=stop_token if stop_token else "",
        lower=lower,
        replace_digit=replace_digit if replace_digit else "")


    if not os.path.exists(file_path):
        rating_field = ntp.dataio.field_reader.DenseVector("rating")
        label_field = ntp.dataio.field_reader.Label(
            "label", vocabulary=["neg", "pos"])
        input_field = ntp.dataio.field_reader.TokenSequence(
            "text", at_least=at_least, start_token=start_token,
            stop_token=stop_token, lower=lower, replace_digit=replace_digit)
        reader = ntp.dataio.file_reader.tsv_reader(
            [input_field, label_field, rating_field], skip_header=True)

        train_data_path = get_imdb_data_path(split="train")
        reader.fit_parameters(train_data_path)

        tr_data_raw = reader.read(train_data_path)
        (tr_inputs, tr_lengths), (tr_labels,), (tr_ratings) = tr_data_raw


        layout = [["inputs", [["sequence", "sequence"], 
                              ["length", "length"]]],
                  ["targets", "targets"]]

        tr_dataset = ntp.dataio.Dataset(
            (tr_inputs, tr_lengths, "sequence"),
            (tr_lengths, None, "length"),
            (tr_labels, None, "targets"),
            layout=layout,
            lengths=tr_lengths)

        tr_file_path = file_path_template.format(
            part="train",
            at_least=at_least,
            start_token=start_token if start_token else "",
            stop_token=stop_token if stop_token else "",
            lower=lower,
            replace_digit=replace_digit if replace_digit else "")

        torch.save({"dataset": tr_dataset, "reader": reader}, tr_file_path)

        te_data_path = get_imdb_data_path(split="test")
        te_data_raw = reader.read(te_data_path)
        (te_inputs, te_lengths), (te_labels,), (te_ratings) = te_data_raw

        te_dataset = ntp.dataio.Dataset(
            (te_inputs, te_lengths, "sequence"),
            (te_lengths, None, "length"),
            (te_labels, None, "targets"),
            layout=layout,
            lengths=te_lengths)

        te_file_path = file_path_template.format(
            part="test",
            at_least=at_least,
            start_token=start_token if start_token else "",
            stop_token=stop_token if stop_token else "",
            lower=lower,
            replace_digit=replace_digit if replace_digit else "")

        torch.save({"dataset": te_dataset, "reader": reader}, te_file_path)

    if not os.path.exists(file_path):
        raise Exception("Failed to create dataset.")

    return torch.load(file_path)

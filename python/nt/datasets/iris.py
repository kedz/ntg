import os
import urllib.request
import json

from nt.datasets import get_data_dir


def download_iris_data():
    ''' 
    Download iris data in four formats:

    iris.header.csv:

    sepal_length,sepal_width,petal_length,petal_width,label
    5.1,3.5,1.4,0.2,Iris-setosa
    .
    .
    .

    iris.no_header.csv: same as above but without header

    iris.feature.label.json -- each line is a separate json object 
                               representing a data point.
                               Each line looks like:
        {"feature": [5.1, 3.5, 1.4, 0.2], "label": "Iris-setosa"}

    iris.features.label.json -- each line looks like
        {"sepal_length": 5.1, "sepal_width": 3.5, 
         "petal_length": 1.4, "petal_width": 0.2,
         "label": "Iris-setosa"}
    '''

    data_dir = get_data_dir()
    url = "https://archive.ics.uci.edu/" \
        "ml/machine-learning-databases/iris/bezdekIris.data"
    response = urllib.request.urlopen(url)
    data = response.read()
    data_string = data.decode("utf-8").strip()

    # Create csv no header version.
    with open(os.path.join(data_dir, "iris.no_header.csv"), "w") as fp:
        fp.write(data_string)

    # Create csv header version.
    with open(os.path.join(data_dir, "iris.header.csv"), "w") as fp:
        header = "sepal_length,sepal_width,petal_length,petal_width,label"
        fp.write(header)
        fp.write("\n")
        fp.write(data_string)

    # Create json version with one feature field containing a list of all
    # four features.
    with open(os.path.join(data_dir, "iris.feature.label.json"), "w") as fp:

        for line in data_string.split("\n"):
            sepal_ln, sepal_wd, petal_ln, petal_wd, label = line.split(",")
            features = [float(sepal_ln), 
                        float(sepal_wd), 
                        float(petal_ln),
                        float(petal_wd)]
            fp.write(json.dumps({"features": features, "target": label}))
            fp.write("\n")

    # Create json version with one field per feature. 
    with open(os.path.join(data_dir, "iris.features.label.json"), "w") as fp:

        for line in data_string.split("\n"):
            sepal_ln, sepal_wd, petal_ln, petal_wd, label = line.split(",")
            ex = {"sepal_length": float(sepal_ln),
                  "sepal_width": float(sepal_wd),
                  "petal_length": float(petal_ln),
                  "petal_width": float(petal_wd),
                  "label": label}
            fp.write(json.dumps(ex))
            fp.write("\n")

def get_iris_data_path(format="json.feature.target"):

    if format == "json.feature.target":
        path = os.path.join(get_data_dir(), "iris.feature.label.json")
    elif format == "json.features.target":
        path = os.path.join(get_data_dir(), "iris.features.label.json")
    elif format == "csv.header":
        path = os.path.join(get_data_dir(), "iris.header.csv")
    elif format == "csv.no_header":
        path = os.path.join(get_data_dir(), "iris.no_header.csv")
    else:
        raise Exception("Invalid format type.")
        
    if not os.path.exists(path):
        download_iris_data()
    return path



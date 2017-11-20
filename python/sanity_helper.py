import urllib.request
import tempfile
import json

def get_iris_data(format):
    url = "https://archive.ics.uci.edu/" \
        "ml/machine-learning-databases/iris/bezdekIris.data"

    if format == "json-feature.target":

        response = urllib.request.urlopen(url)
        data = response.read()
        text_data = data.decode("utf-8").strip()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            for line in text_data.split("\n"):
                
                sepal_ln, sepal_wd, petal_ln, petal_wd, label = line.split(",")
                features = [float(sepal_ln), 
                            float(sepal_wd), 
                            float(petal_ln),
                            float(petal_wd)]
                fp.write(json.dumps({"features": features, "target": label}))
                fp.write("\n")
            return fp.name
    else:
        raise Exception()

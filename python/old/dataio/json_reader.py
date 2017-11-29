import json

class JSONReader:
    def __init__(self, readers):
        self.readers_ = readers

    def fit_parameters(self, path):
        self.apply_readers(path) 
        for reader in self.readers_:
            reader.fit_parameters()
            reader.reset_saved_data()

    def apply_readers(self, path):
        with open(path, "r") as fp:
            for line in fp:
                datum = json.loads(line)
                for reader in self.readers_:
                    reader.read(datum)

    def read(self, path):
        self.apply_readers(path)
        all_reader_data = []
        for reader in self.readers_:
            all_reader_data.append(reader.finish_read())
        return all_reader_data

def collect_json_stats(path, readers, skip_header=True, diagnostics=True):

    pass

def apply_json_readers(path, readers):
    with open(path, "r") as fp:
        for line in fp:
            datum = json.loads(line)
            for reader in readers:
                reader.read(datum)

    all_reader_data = []
    for reader in readers:
        all_reader_data.append(reader.finish())

    return all_reader_data

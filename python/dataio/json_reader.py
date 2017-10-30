import json

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

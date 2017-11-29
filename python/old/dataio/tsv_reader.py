

def collect_tsv_stats(path, readers, skip_header=True, diagnostics=True):

    with open(path, "r") as fp:
        if skip_header:
            fp.readline()
        for line in fp:
            data = line.split("\t")
            for reader in readers:
                reader.collect_stats(data)

    for reader in readers:
        reader.freeze_vocab()

    if diagnostics:
        for reader in readers:
            print(reader.info())

def apply_tsv_readers(path, readers, skip_header=True):

    with open(path, "r") as fp:
        if skip_header:
            fp.readline()
        for line in fp:
            data = line.split("\t")
            for reader in readers:
                reader.read(data)

    all_reader_data = []
    for reader in readers:
        all_reader_data.append(reader.finish())

    return all_reader_data

from .file_reader_base import FileReaderBase
import json

class JSONReader(FileReaderBase):
    def __init__(self, readers, line_separated=True, verbose=False):
        super(JSONReader, self).__init__(readers, verbose=verbose)
        self.line_separated = line_separated

    def apply_readers(self, path):

        with open(path, "r") as fp:
            for reader in self.readers_:
                reader.clear_field_map()

            if self.line_separated:
                
                for line in fp:
                    data = json.loads(line)
                    for reader in self.readers:
                        reader.read(data)
            else:
                dataset = json.load(fp)
                for data in dataset:
                    for reader in self.readers:
                        reader.read(data)

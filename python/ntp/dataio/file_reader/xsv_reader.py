from .file_reader_base import FileReaderBase
import sys


class XSVReader(FileReaderBase):
    def __init__(self, readers, sep, skip_header=False, verbose=False,
                 show_progress=False):
        super(XSVReader, self).__init__(readers, verbose=verbose)
        self.sep_ = sep
        self.skip_header_ = skip_header
        self.show_progress = show_progress

    @property
    def sep(self):
        return self.sep_

    @property
    def skip_header(self):
        return self.skip_header_

    @skip_header.setter
    def skip_header(self, skip_header):
        self.skip_header_ = skip_header

    def apply_readers(self, path):
        read_lines = 0
        with open(path, "r") as fp:
            if self.skip_header:
                header_dict = {}
                header = fp.readline().strip().split(self.sep)

                for i, field in enumerate(header):
                    header_dict[field] = i
                    if self.verbose:
                        print("xsv header {}: {}".format(i, field))

                for reader in self.readers:
                    if reader.field_type == str:
                        reader.update_field_map(header_dict[reader.field])
            else:
                for reader in self.readers_:
                    reader.clear_field_map()

            for line in fp:
                read_lines += 1
                if self.show_progress:
                    sys.stdout.write(
                        "\rread {} lines".format(read_lines))
                    sys.stdout.flush()

                items = line.strip().split(self.sep)
                for reader in self.readers:
                    reader.read(items)

            if self.show_progress:
                print("")

def tsv_reader(readers, **kwargs):
    return XSVReader(readers, "\t", **kwargs)

def csv_reader(readers, **kwargs):
    return XSVReader(readers, ",", **kwargs)

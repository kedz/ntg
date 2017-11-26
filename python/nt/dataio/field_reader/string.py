from .field_reader_base import FieldReaderBase


class String(FieldReaderBase):

    def __init__(self, field):
        super(String, self).__init__(field)
        self.register_data("string_store")

    def read_extract(self, data):
        if not isinstance(data, str):
            data = str(data)
        self.string_store.append(data)


    def finalize_saved_data(self):
        return (tuple(self.string_store),)

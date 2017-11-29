from dataio.reader.reader_base import ReaderBase2

class MetaDataStore(ReaderBase2):

    def __init__(self, fields):
        super(MetaDataStore, self).__init__()
        self.fields_ = tuple(fields)
        self.register_data("metadata")
    
    @property
    def fields(self):
        return self.fields_

    def read(self, datum):
        md = {}
        for field in self.fields: 
            md[field] = datum[field]
        self.metadata.append(md)

    def finalize_saved_data(self):
        data = (self.metadata,)
        return data

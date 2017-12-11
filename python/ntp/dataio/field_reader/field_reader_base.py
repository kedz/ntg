from abc import ABC, abstractmethod

class FieldReaderBase(ABC):
    def __init__(self, field):
        super(FieldReaderBase, self).__init__()
        self.data_attributes_ = []
        
        if not isinstance(field, (str, int, type(None))):
            raise Exception("field must be of type str or int.")

        self.field_ = field
        self.field_type_ = type(field)
        self.field_dict_ = {}

        self.field_map_ = None

    @property
    def field_map(self):
        return self.field_map_

    def update_field_map(self, map_value):
        
        if self.field_type == int:
            raise Exception("Cannot set map value for integer field.")

        if not isinstance(map_value, int):
            raise Exception("map_value must be of type int.")
        self.field_map_ = map_value

    def clear_field_map(self):
        self.field_map_ = None

    @property
    def field(self):
        return self.field_

    @property
    def field_type(self):
        return self.field_type_

    def register_data(self, name):
        self.data_attributes_.append(name)
        setattr(self, name, [])

    def reset_saved_data(self):
        for name in self.data_attributes_:
            setattr(self, name, [])
 
    def read(self, raw_instance):
        if self.field_map is not None:
            return self.read_extract(raw_instance[self.field_map])
        elif self.field is not None:
            return self.read_extract(raw_instance[self.field])
        else:
            self.read_extract(raw_instance)

    @abstractmethod
    def read_extract(self, data):
        pass

    def fit_parameters(self):
        pass

    @abstractmethod
    def finalize_saved_data(self):
        pass

    def finish_read(self, reset=True):
        data = self.finalize_saved_data()
        if reset:
            self.reset_saved_data()
        return data

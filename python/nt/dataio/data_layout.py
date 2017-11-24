from collections import namedtuple

# TODO make work with dict layout and make todict use ordered dict

class DataLayout(object):
    def __init__(self, layout_meta, label2data, root_name="dataset"):
        
        self.layout_meta_ = layout_meta
        self.label2data_ = label2data
        self.root_name_ = root_name
        self.replacement_sites_ = {} # TODO can probably remove this
        self.layout_ = self.recursive_layout_init_(layout_meta, root_name)
        
    def recursive_layout_init_(self, ld, name):
        attributes = []
        values = []
        for key, value in ld:
            if isinstance(value, (list,tuple)):
                attributes.append(key)
                ds = self.recursive_layout_init_(value, key)
                values.append(ds)
            else:
                attributes.append(key)
                values.append(self.label2data_[value])
        return namedtuple(name, attributes)(*values) 

    def __iter__(self):
        for item in self.layout_:
            yield item

    def __getitem__(self, item):
        return self.layout_[item]

    def index_select(self, index):
        idx_label2data = {}
        for label, data in self.label2data_.items():
            idx_label2data[label] = data.index_select(0, index)
        return DataLayout(
            self.layout_meta, idx_label2data, root_name=self.root_name_)

    def to_dict_helper_(self, t):
        
        if hasattr(t, "_fields"):
            result = {}
            for field in t._fields:
                result[field] = self.to_dict_helper_(getattr(t, field))
            return result
        else:
            return t

    def to_dict(self):
        d = {}
        for field in self.layout_._fields:
            d[field] = self.to_dict_helper_(getattr(self.layout_, field))
        return d

    @property
    def label2data(self):
        return self.label2data_

    @property
    def layout_meta(self):
        return self.layout_meta_

    def __getattr__(self, key):
        return getattr(self.layout_, key)

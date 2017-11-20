import os
import sanity_helper

import nt



import torch
from models import MLPClassifier
from dataio.json_reader import JSONReader
from dataio.reader.label_reader import LabelReader2
from dataio.reader.dense_real_vector_reader import DenseRealVectorReader
from dataset import Dataset3


import optimizer
import criterion


import trainer
import crossval_trainer

def main():
    data_path = sanity_helper.get_iris_data("json-feature.target")

    feature_reader = DenseRealVectorReader.field_reader("dict", "features", 4)
    target_reader = LabelReader2.field_reader("dict", "target")
    data_reader = JSONReader([feature_reader, target_reader])
    data_reader.fit_parameters(data_path)
    (features,), (targets,) = data_reader.read(data_path)
   
    dataset = Dataset3((features, "inputs"), (targets, "targets"),
                       batch_size=1, shuffle=True, gpu=-1)
    

    tr_idx, val_idx, te_idx = nt.trainer.stratified_generate_splits(
        [i for i in range(dataset.size)], dataset.targets)

    train_dataset = dataset.index_select(tr_idx)
    valid_dataset = dataset.index_select(val_idx)



    model = MLPClassifier(4, 3)
    opt = optimizer.Adam(model.parameters(), lr=.01)

    crit = nt.criterion.BinaryCrossEntropy(mode="logit")
        #class_names=target_reader.labels)
    
    trainer.minimize_criterion(crit, train_dataset, valid_dataset, 5)
    #crossval_trainer.minimize_criterion(crit, dataset, 5, stratify=False)


     

    os.remove(data_path)

if __name__ == "__main__":
    main()

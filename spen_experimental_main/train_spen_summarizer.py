import argparse
import dataio
from dataio.recipes.summarization import read_mds_embedding_data
from dataio.reader.mds_embedding_reader import MDSEmbeddingReader

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)

    args = parser.parse_args()
    
    reader = MDSEmbeddingReader()
    data_train = read_mds_embedding_data(
        args.train, reader, batch_size=2, gpu=-1)

    for batch_num, batch in enumerate(data_train.iter_batch()):
        print("Batch {}".format(batch_num))
        for b, texts in enumerate(batch.texts):
            for i, text in enumerate(texts[:10]):
                print(batch.targets.data[b,i], text) 
                print(batch.inputs.data[b,i:i+1,:9])
        print("")




if __name__ == "__main__":
    main()

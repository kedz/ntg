data_path=/proj/nlp/users/chris/ntg_proj/datasets/parallel-content/
train_path=${data_path}/0-130k.wp.swp.content.train.tsv
valid_path=${data_path}/0-130k.wp.swp.content.valid.tsv
test_path=${data_path}/0-130k.wp.swp.content.test.tsv

export PYTHONPATH=$PYTHONPATH:/proj/nlp/users/chris/ntg_proj/orca/ntg/python

#PYTHONPATH=$PYTHONPATH python trainer_main.py \
#    --train $train_path \
#    --valid $valid_path \
#    --gpu 0 \
#    --attention dot \
#    --enc-vocab-size 80000 \
#    --dec-vocab-size 40000 \
#    --embedding-dim 300 \
#    --hidden-dim 300 \
#    --rnn gru \
#    --num-layers 1 \
#    --bidirectional 1 \
#    --dropout 0 \
#    --optimizer adagrad \
#    --lr .001 \
#    --epochs 50 \
#    --batch-size 16 \
#    --length-mode decoder-input-embedding1 \
#    --save-path ../../models/lt1.1.mdl 

#PYTHONPATH=$PYTHONPATH python trainer_main.py \
#    --train $train_path \
#    --valid $valid_path \
#    --gpu 0 \
#    --attention dot \
#    --enc-vocab-size 80000 \
#    --dec-vocab-size 40000 \
#    --embedding-dim 300 \
#    --hidden-dim 300 \
#    --rnn gru \
#    --num-layers 1 \
#    --bidirectional 1 \
#    --dropout 0 \
#    --optimizer adam \
#    --lr .0001 \
#    --epochs 50 \
#    --batch-size 16 \
#    --length-mode decoder-input-embedding1 \
#    --save-path ../../models/lt1.2.mdl 

PYTHONPATH=$PYTHONPATH python trainer_main.py \
    --train $train_path \
    --valid $valid_path \
    --gpu 0 \
    --attention dot \
    --enc-vocab-size 80000 \
    --dec-vocab-size 40000 \
    --embedding-dim 300 \
    --hidden-dim 300 \
    --rnn rnn \
    --num-layers 1 \
    --bidirectional 1 \
    --dropout 0 \
    --optimizer adam \
    --lr .0001 \
    --epochs 50 \
    --batch-size 16 \
    --length-mode decoder-input-embedding1 \
    --save-path ../../models/lt1.3.mdl 




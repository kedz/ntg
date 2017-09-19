
data_path=/proj/nlp/users/chris/ntg_proj/datasets/parallel-content/
train_path=${data_path}/0-130k.wp.swp.content.train.tsv
valid_path=${data_path}/0-130k.wp.swp.content.valid.tsv
test_path=${data_path}/0-130k.wp.swp.content.test.tsv


export PYTHONPATH=$PYTHONPATH:/proj/nlp/users/chris/ntg_proj/orca/ntg/python

PYTHONPATH=$PYTHONPATH python reglen.py \
    --model dummy.mdl \
    --train $train_path \
    --valid $valid_path \
    --test $test_path
 

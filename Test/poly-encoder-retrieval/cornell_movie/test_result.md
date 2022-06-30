transformer/polyencoder
===============

training a new retrieval cross-encoder model with cornell_movie dataset

`!parlai train_model --init-model zoo:pretrained_transformers/cross_model_huge_reddit/model \
-t cornell_movie --dict-file zoo:pretrained_transformers/cross_model_huge_reddit/model.dict \
--model transformer/crossencoder --batchsize 1 --eval-batchsize 10 --eval-candidates batch \
--warmup_updates 1000 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
-lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
--text-truncate 360 --num-epochs 12.0 --max_train_time 200000 -veps 0.25 \
-vme 2500 --validation-metric accuracy --validation-metric-mode max \
--save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
--dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
--variant xlm --reduction-type first --share-encoders False \
--learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
--attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
--embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
--learn-embeddings True --dict-endtoken __start__ \
--model-file retrieval_model/retrieval_model`

problem
===============


Link
===============

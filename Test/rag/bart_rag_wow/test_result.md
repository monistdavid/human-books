rag_wow
===============
`parlai train_model --model rag --task wizard_of_wikipedia \
--rag-model-type token --rag-retriever-type dpr --dpr-model-file zoo:hallucination/multiset_dpr/hf_bert_base.cp \
--generation-model bart --init-opt arch/bart_large --add-missing-turns train \
--batchsize 1 --fp16 True --gradient-clip 0.1 --label-truncate 128 \
--log-every-n-secs 30 --lr-scheduler reduceonplateau --lr-scheduler-patience 1 \
--model-parallel True --optimizer adam --text-truncate 512 --truncate 512 \
--learningrate 1e-05 --validation-metric-mode min --validation-every-n-epochs 0.01 \
--validation-max-exs 1000 --validation-metric ppl --validation-patience 5 \
--model-file bart_rag_wow/trained_bart_rag`


very bad performance
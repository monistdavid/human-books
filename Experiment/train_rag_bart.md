#Finetune blenderbot-400M-distill with dog dialogues (Huggingface) .

`!parlai train_model --model rag --task cmu_dog \
--rag-model-type token --rag-retriever-type dpr --dpr-model-file zoo:hallucination/multiset_dpr/hf_bert_base.cp \
--generation-model bart --init-opt arch/bart_large \
--batchsize 1 --fp16 True --gradient-clip 0.1 --label-truncate 128 \
--log-every-n-secs 30 --lr-scheduler reduceonplateau --lr-scheduler-patience 1 \
--model-parallel True --optimizer adam --text-truncate 512 --truncate 512 \
--learningrate 1e-05 --validation-metric-mode min --validation-every-n-epochs 0.01 \
--validation-max-exs 1000 --validation-metric ppl --validation-patience 5 \
--model-file bart_rag/trained_bart_rag`

1. do not train multiple not similar dataset together, this will ruin the model badly.
2. 




Thoughts
===============
human can learn multiple things in a period of time. Or at least they can learn multiple things as a human
being. What should model to improve itself.
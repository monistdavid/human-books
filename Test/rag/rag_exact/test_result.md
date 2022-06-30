rag_exact_index
===============
bard_rag_sequence model with custom faiss index file. 

generate dense embeddings
`!python /usr/local/lib/python3.7/dist-packages/parlai/agents/rag/scripts/generate_dense_embeddings.py -mf zoo:hallucination/multiset_dpr/hf_bert_base.cp --dpr-model True \
--passages-file DPR/dog_data/dog_index.csv --outfile DPR/embeddings \
--shard-id 0 --num-shards 1 -bs 1`

generate faiss index
`!python /usr/local/lib/python3.7/dist-packages/parlai/agents/rag/scripts/index_dense_embeddings.py --retriever-embedding-size 768  \
--embeddings-dir DPR/embeddings --embeddings-name embeddings --indexer-type exact --compressed-indexer-factory HNSW32
`
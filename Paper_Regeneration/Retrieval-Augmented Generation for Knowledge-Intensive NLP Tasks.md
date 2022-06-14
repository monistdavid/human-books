Link    
===============
<p>

https://arxiv.org/pdf/2005.11401v4.pdf

https://parl.ai/docs/agent_refs/rag.html

</p>

Train   
===============
1. obtain a DPR model
   1. from DPR repository: https://github.com/facebookresearch/DPR
   2. from parlai zoo: zoo:hallucination/multiset_dpr/hf_bert_base.cp
2. generate Dense Embeddings with the dataset: 
   integer document id starting at zero<tab>document text<tab>document title
3. index the Dense Embeddings


Question   
===============
1. virtual void faiss::Clustering::train(faiss::Clustering::idx_t, const float*, faiss::Index&) 
   at /home/logidliec/dev/faiss/Clustering.cpp:68:
   Error: 'nx >= k' failed: need at least as many training points as clusters
   1. Note the default index factory setting is IVF4096_HNSW128,PQ128, if you are processing small files, 
      you may encounter errors such as Error: 'nx >= k' failed, then you need to set 
     --compressed-indexer-factory to other indexes in the index factory in FAISS such as HNSW32.
2. FileNotFoundError: Could not find filename 'bart_large or opt preset 'bart_large.opt'. 
   Please check https://parl.ai/docs/opt_presets.html for a list of available opt presets.






Thought
===============
3. 测试论文效果不好的主要原因可能是因为数据库的混乱和不合适。我们使用的MSC数据集更多针对的是人而不是狗。
Link    
===============
<p>

https://arxiv.org/pdf/2107.07566v1.pdf

</p>


Notes
===============
1. We find that search-query based access of the internet in conversation provides
   superior performance compared to existing approaches that either use no augmentation or FAISS-based retrieval.
2. The retrieval system uses a DPR (Dense Passage Retrieval) (Karpukhin et al., 2020) Transformer-based model
    which scores document-context pairs in order to rank them based on their match using a bi-encoder
    framework, where the base DPR model is pretrained on QA data pairs.
3. It has been observed before that large language models, when augmented with retrieval, have trouble with 
   choosing between copying knowledge remembered within their weights and knowledge provided in 
   retrieved documents (Shuster et al., 2021). Here, we propose a general regularization method 
   to more finely control this mechanism: when training, we multi-task between the original response 
   generation task and a new task which consists of generating the selected knowledge from retrieved 
   documents indicated by human annotators3 . The second task can be seen as a regularizer that encourages 
   the use of retrieved documents, as the easiest way for the model to do well on that task is to attend 
   and copy to the document where that text already exists. Then, by changing the mixing parameter 
   between the two tasks, the intent is to achieve a smooth control between encouraging copying from
   retrieved documents, or not.



Thoughts
===============



Summary   
===============
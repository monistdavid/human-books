Link    
===============
<p>

https://arxiv.org/pdf/2104.07567.pdf

</p>


Notes
===============
1. RAG: The technique employs an encoder-decoder to encode the
   question and decode (generate) the answer, where the encoding is augmented with documents 
   or passages retrieved from a large unstructured document set using a learnt matching 
   function; the entire neural network is typically trained end-to-end.
2. determining which specific elements of a given piece of knowledge are informative to the
   dialogue, which is commonly referred to as “knowledge selection”
3. learning how to attend to the relevant knowledge
4. RAG and FiD
5. Seq2seq Models
   1. BART
   2. T5
   3. BlenderBot
6. DPR, as a bi-encoder architecture, transforms both sequences independently into fixed length vectors,
   and thus limits the interaction between a dialogue context and a candidate document to a final dotproduct similarity score. However, allowing more
   interaction between a context and candidate yields superior results in various information retrieval and
   ranking tasks. 
7. Retrieval Improving
   1. DPR-Poly
   2. PolyFAISS
   3. ColBERT
   4. ReGReT
   5. BREAD
   6. TREAD
8. Improving Augmented Generation
   1. RAG-Turn




Thoughts
===============
As author mentioned, the retrieval document could possibly be used as long-term memory. What
should be the short term memory? The blenderbot itself? What is the core difference between long-term and short-term
memory?


Summary
===============
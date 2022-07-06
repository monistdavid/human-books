Link    
===============
<p>

https://arxiv.org/pdf/2203.13224.pdf

</p>


Notes
===============
1. aggregating information from multiple retrieved documents is a difficult problem, which may result in
   incorporating parts of multiple documents into one factually incorrect response.
2. The first step, given the input context, generates a relevant search query for an internet search engine,
   while the second step is fed the returned documents and generates their most relevant portion. The last
   step uses that knowledge to produce its final response. By decomposing this difficult problem into
   three manageable steps, pertinent up-to-date information can be incorporated into the final language
   model generation.
3. BB2 grounds on retrieval from the internet for open-domain dialogue tasks, but does not use a modular approach to
   generate knowledge, instead applying the fusion-indecoder (FiD) method 
   to output a response directly given the retrieved documents.
4. SeeKeR consists of three modules
   1. Search Module Given the encoded input context, a search query is generated. 
   2. Knowledge Module Given the encoded input context, and a set of retrieved documents, a knowledge 
      response is generated.
   3. Response Module Given the encoded input context concatenated with the knowledge response,
      the final response is generated.



Thoughts
===============



Summary   
=============== 
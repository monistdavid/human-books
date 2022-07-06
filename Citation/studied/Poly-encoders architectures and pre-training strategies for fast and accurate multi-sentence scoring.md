Link    
===============
<p>

https://arxiv.org/pdf/1905.01969.pdf

</p>


Notes
===============
1. For tasks that make pairwise comparisons between sequences, matching a given input with a
   corresponding label, two approaches are common: Cross-encoders performing full self-attention over the
   pair and Bi-encoders encoding the pair separately. The former often performs
   better, but is too slow for practical use. In this work, we develop a new transformer architecture, 
   the Poly-encoder, that learns global rather than token level self-attention features. 
2. Bi-encoders and Cross-encoders. Cross-encoders, which perform full (cross)
   self-attention over a given input and label candidate, tend to attain much higher accuracies than their
   counterparts, Bi-encoders, which perform self-attention over the input and candidate label separately 
   and combine them at the end for a final representation. As the representations are separate, 
   Bi-encoders are able to cache the encoded candidates, and reuse these representations for each input 
   resulting in fast prediction times. Cross-encoders must recompute the encoding for each input and 
   label; as a result, they are prohibitively slow at test time.







Thoughts
===============



Summary
===============
Link
===============
<p>

https://arxiv.org/pdf/1911.00536.pdf

</p>


Notes
===============

1. GPT2 have the capacity to capture textual data with fine granularity and produce output with a high-resolution that
   closely emulates real-world text written by humans.
2. Like GPT-2, DIALOGPT is formulated as an autoregressive (AR) language model, and uses the multi-layer transformer as
   model architecture. Unlike GPT-2, however, DIALOGPT is trained on large-scale dialogue pairs/sessions extracted from
   Reddit discussion chains.
3. The dataset is extracted from comment chains scraped from Reddit spanning from 2005 till 2017. Reddit discussions can
   be naturally expanded as tree-structured reply chains, since a thread replying to one thread forms the root node of
   subsequent threads. We extract each path from the root node to the leaf node as a training instance containing
   multiple turns of dialogue. We filter the data by removing the instances where (1) there is a URL in source or
   target, (2) where the target contains word repetitions of at least three words, (3) where the response does not
   contain at least one of the top-50 most frequent English words (e.g., “the”, “of”, “a”), since this probably
   indicates it might not be an English sentence, (4) where the response contains special markers such as “[” or “]”, as
   this could be markup language, (5) where source and target sequences together are longer than 200 words, (6) where
   the target contains offensive language, identified by phrase matching against a large blocklist. We also excluded a
   large number of subreddits that had been identified as likely to contain offensive content. In addition, we
   aggressively filtered out blandness, e.g., removing instances where the responses contained 90% of tri-grams that
   have been seen more than 1000 times. Often uninformative, such responses account for about 1% of the data. After
   filtering, the dataset comprises 147,116,725 dialogue instances, in total 1.8 billion words.
4. We trained our DIALOGPT model on the basis of the GPT-2 (Radford et al., 2018) architecture.The GPT-2 transformer
   model adopts the generic transformer language model (Vaswani et al., 2017)
   and leverages a stack of masked multi-head selfattention layers to train on massive web-text data.
5. Our model inherits from GPT-2 (Radford et al., 2018), a 12-to-48 layer transformer with layer normalization, a
   initialization scheme that accounts for model depth that we modified, and byte pair encodings (Sennrich et al., 2016)
   for the tokenizer.We follow the OpenAI GPT-2 to model a multiturn dialogue session as a long text and frame the
   generation task as language modeling. We first concatenate all dialog turns within a dialogue session into a long
   text x1, · · · , xN (N is the sequence length), ended by the end-of-text token. We denote the source sentence (
   dialogue history)as S = x1, · · · , xm and target sentence (ground truth response) as T = xm+1, · · · , xN , the
   conditional probability of P(T|S) can be written as the product of a series of conditional probabilities:
   ![img_7.png](img_7.png) For a multi-turn dialogue session T1, · · · , TK, (1)
   can be written as p(TK, · · · , T2|T1), which is essentially the product of conditional probabilities of p(Ti |T1, ·
   · · , Ti−1). Consequently, optimizing a single objective p(TK, · · · , T2|T1) can be perceived as optimizing all p(Ti
   |T1, · · · , Ti−1)
   source-target pairs.
6. Maximum mutual information scoring function

Thoughts
===============

1. DIALOGPT is trained on 147M conversation-like exchanges from reddit?
    1. why not directly train on conversational dialogue?
2. What kind of information or abilities does language models have?
    1. Does language model understand words relationships or sentence relationships? For example, if I have context
       about Harry Potter, does the language models generate sentences related to harry potter in both the language
       style and the related context relationship? Or it can only generate a making sense sentences.
3. joint event in probability depends on two variables. Joint probability represent the probability that both of the
   variables appear, what is the probability of it while all the other possibilities sum up to 1.
   ![](img.png)
   joint probability distribution means all the possibilities that includes certain two variables that sum up equal to 1
   ![img_1.png](img_1.png)
4. marginal probability (simple probability)
   ![img_2.png](img_2.png)
   ![img_3.png](img_3.png)
   ![img_4.png](img_4.png)
5. all probability distribution sum up to 1
6. probability distribution tells the probability of certain variables in the whole environment.
7. ![img_5.png](img_5.png)
   ![img_6.png](img_6.png) flip two coins, fliping coins are independent events, the probability of getting both coins
   head is 0.5 x 0.5 which is 0.25, the reason why we can do it is because they are independent to each other.
8. papers always say they release the source code at the end of the introduction. Why?
    1. People will be interested in open sourced paper
    2. people will get interested in papers more after they get interested in the concept
9. How do I know if the parameters of a model is large enough for the training purpose.

Summary
===============
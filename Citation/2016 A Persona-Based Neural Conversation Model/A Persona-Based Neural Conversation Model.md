Link
===============
<p>

https://arxiv.org/pdf/1603.06155.pdf

</p>


Notes
===============

1. A persona can be viewed as a composite of elements of identity (background facts or user profile), language behavior,
   and interaction style.
2. The Speaker Model integrates a speaker-level vector representation into the target part of the SEQ2SEQ model.
   Analogously, the Speaker-Addressee model encodes the interaction patterns of two interlocutors by constructing an
   interaction representation from their individual embeddings and incorporating it into the SEQ2SEQ model. These
   persona vectors are trained on human-human conversation data and used at test time to generate personalized responses
3. The SMT model proposed by Ritter et al., on the other hand, is end-to-end, purely data-driven, and contains no
   explicit model of dialog structure; the model learns to converse from human-to-human conversational corpora.
4. Again, the model uses seq-to-seq models
   ![img.png](img.png)
   ![img_1.png](img_1.png)
5. Our work introduces two persona-based models:
   the Speaker Model, which models the personality of the respondent, and the Speaker-Addressee Model which models the
   way the respondent adapts their speech to a given addressee — a linguistic phenomenon known as lexical entrainment (
   Deutsch and Pechmann, 1982).

Thoughts
===============

1. It is really hard to mimic a human's talking behavior, but that is not important because a book character's
   personality is not complete as well in the book, it is extremely hard to completely regenerate the whole character.
   the key point of showing a book character is to by speaking out its story vividly.
2. By integrating persona improves BLEU score, why?
    1. because there are a lot of hidden properties under the dialogues, by explicitly integrating those properties
       could probabliy improve the performance of chatbot.
3. inputs and outputs use different LSTMs with separate parameters to capture different compositional patterns.
    1. what does this mean?
4.

Summary
===============

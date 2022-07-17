Link
===============
<p>

https://aclanthology.org/2021.emnlp-main.183.pdf

</p>


Notes
===============

1. Generally, grounded dialogue modeling trains a dialogue model on a dataset D that consists of triples (c, r, g),
   where c is the dialogue history, r is the response, and g is the grounded concept. The model is generally optimized
   using maximum likelihood estimate (MLE)
   ![img.png](img.png)
   Despite its effectiveness, this formulation faces two challenges regarding transferability.
    1. grounded dialogue datasets are usually collected under a guided setting, e.g., annotators are usually encouraged
       to embed personaor knowledge into responses, which leads to a distributional gap between the conversations in a
       grounded dialogue dataset and natural conversations. As a result, models trained with Eq. (1) may generate
       unnatural responses and are vulnerable to the distributional shift of the dialogue history.
    2. at inference time, models trained with Eq. (1) cannot be grounded on unseen types of concept g 0 other than g. An
       example for such grounding gap is that a model trained on PERSONACHAT (Zhang et al., 2018) with Eq. (1)
       cannot be grounded on world knowledge.
    3. To address the above transferability challenges, we propose a grounded minimal editing framework for grounded
       dialogue modeling. Instead of learning a grounded response generator as is done in Eq. (1), we propose to learn a
       grounded minimal editor that operates on existing responses. Specifically, suppose we have an original response r
       o that is coherent with the dialogue history c but is not grounded on the concept g. Our goal is to minimally
       edit r o such that it is grounded on the concept g and coherent with the dialogue history c.
    4. Note that collecting paired responses before and after editing is resource-consuming; thus, our goal is to learn
       the editing without paired data.
    5. ![img_1.png](img_1.png)
2. Main contributions
    1. We propose a framework named grounded minimal editing to address the transferability challenges of grounded
       dialogue modeling.
    2. We propose Grounded Minimal Editor (GME)
       and present the PERSONAMINEDIT dataset to evaluate GME’s effectiveness for personagrounded minimal editing.
    3. Experimental results show that GME largely outperforms strong baselines on the PERSONAMINEDIT dataset. GME is
       also transferable to edit other models’ outputs and improve the persona consistency while preserving their use of
       knowledge and empathy.
3. What stands out in this paper:
    1. try to solve the two main transferability challenges
    2. essentially different from existed idea of editing existing responses:
        1. Regarding the formulation, we emphasize minimal editing, while previous works do not. As analyzed in Section
           1, minimal editing is an important component to address the transferability challenges.
        2. Regarding the training algorithm, previous works derive templates from self-generated or retrieved texts,
           while our model derives templates from the observed responses.
4. Formulation
    1. ![img_2.png](img_2.png)
    2. Grounded dialogue modeling uses a dataset D that consists of triples (c, r, g), where c, r, and g are the
       dialogue history, the response, and the grounded concept, which are shown in grey in the left part of Figure 2.
       To formulate the term “minimal”, we need to add unobserved variables into the graphical model, denoted as u in
       Figure 2, which cover all unobserved variables. The graph states that r = f(c, g, u). As shown in the right part
       of Figure 2, we observe (c, ro , ge
       ) at inference time, where r o and g e stand for the original response and the grounded concept for editing. The
       graph states that the original response r o = f(c, go , u), where g o represents the concept the original
       response is grounded on, and that both g o and u are unobserved. The edited response is defined as r e = f(c, ge
       , u), which replaces g o as g e , and keeps c and u intact. Our formulation follows the idea of counterfactual
       reasoning (Peters et al., 2017), and it guarantees that 1) the content irrelevant to the grounded concept is
       preserved, and that 2) the edited response is coherent with the dialogue history. Since it is costly to collect
       paired (r o , re
       ) for training, the grounded minimal editor should be trained on the grounded dialogue data (c, r, g) ∼ D as in
       Eq. (1). As the first attempt toward the proposed framework, we focus on persona-grounded minimal editing in the
       experiments. Thus, in the remaining part of this paper, we set the grounded concept g, g o , g e as the persona
       p, p o , p e .
5. Approach
    1. At inference, GME first creates a response template t by masking persona-related spans in the original response r
       o and then recombines the template t, the persona p e , and the dialogue history c into an edited response r e .
       ![img_3.png](img_3.png)
    2. Recombination Module
        1. The recombination module learns to recombine the response template, the persona, and the dialogue history as
           the edited response.
        2. Span mask
           1. The span mask serves as the placeholder of persona-related spans.

Thoughts
===============

1. the first question mentioned in the challenges regarding transferability also appeared in the RAG, sometimes the
   model tend to directly copy and paste the information from the documents. This paper names this issue distributional
   gap.
2. Are all the model problems caused by integrating? If we further divided a model's task to several subtasks, seems
   like the model could perform better.
3. We all know the data is the key for any DL model. What would I do if there is no data available? What is the key idea
   for solving problem that does not have enough data supporting. (no parallel data)

Summary
===============
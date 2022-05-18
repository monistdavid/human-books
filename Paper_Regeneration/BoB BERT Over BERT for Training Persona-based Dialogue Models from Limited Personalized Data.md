Link    
===============
<p>

https://arxiv.org/pdf/2106.06169.pdf

https://github.com/songhaoyu/BoB

</p>

Train   
===============
1. with MSC (Multiple Session Chat) dataset (including context length between 6 - 14.
Below is just a short example of MSC data). The data includes a personality, a context
and a response<br>
   1. "personas": ["I don't own a car. I enjoy running and walking. 
      I live in a small town. I live semi-close towkr.",
      "I'm a computer programmer.", "I like grilling steak.", 
      "I am from Alaska. I like wearing warm pants."]
   2. 
      1. I need some advice on where to go on vacation, have you been anywhere lately?	
      2. I have been all over the world. I'm military.
   3. 
      1. I need some advice on where to go on vacation, have you been anywhere lately?	
      2. Yes! I am an eager runner, so my clothes get sweaty a lot.
2. with MNLI (entailment & contradiction) dataset(this is used to train to recognize if the 
following sentence is having similar meaning as the previous one or completely different.
   1. https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip (entailment & contradiction)


Question   
===============

Thought
===============
1. 测试论文效果不好的主要原因可能是因为数据库的混乱和不合适。我们使用的MSC数据集更多针对的是人而不是狗。
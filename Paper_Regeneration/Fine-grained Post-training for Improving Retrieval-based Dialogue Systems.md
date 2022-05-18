Link    
===============
<p>

https://aclanthology.org/2021.naacl-main.122.pdf

https://github.com/hanjanghoon/BERT_FP

</p>


Train   
===============
1. with dog data(start with 1, 0 to represent the gold response or not, then start with dog speech
/person speech/dog speech/person speech/response, the length of context is fixed as 4)<br>
    1. 1. 1	
       2. All Dogs go to heaven	
       3. You seem confident	
       4. When I put my front paws together I fall on my face	
       5. What does that mean?	
       6. It means IT'S ANATOMICALLY IMPOSSIBLE for a Dog to pray<br>
    2. 1. 0	
       2. All Dogs go to heaven
       3. You seem confident
       4. When I put my front paws together I fall on my face
       5. What does that mean?
       6. Dogs have sound phobia for noises that they are not familiar with.
2. with Cornell data(start with 1, 0 to represent the gold response or not, then start with 
person 1 speech/person 2 speech..., the length of context is not fixed. Below is just a 
short example of Cornell movie data)<br>
   1. 
      1. 1	
      2. you got something on your mind	
      3. i counted on you to help my cause you and that thug are obviously 
      failing arent we ever going on our date
   2. 
      1. 0	
      2. you got something on your mind	
      3. separate incidents
3. with cornell data and dog data at the end of dataset)<br>
4. with MSC (Multiple Session Chat) dataset (including context length between 6 - 14.
Below is just a short example of MSC data)<br>
   1. 
      1. 1	
      2. I need some advice on where to go on vacation, have you been anywhere lately?	
      3. I have been all over the world. I'm military.
   2. 
      1. 0	
      2. I need some advice on where to go on vacation, have you been anywhere lately?	
      3. Yes! I am an eager runner, so my clothes get sweaty a lot.
6. with MSC (Multiple Session Chat) dataset (only including several fixed context length, 13, 21, 25)<br>
7. with MSC (Multiple Session Chat) dataset (including context length between 2 - 19) <br>


Question   
===============
1. TypeError: object of type 'float' has no len()， ValueError: array split does not result in an equal division. （已解决）
    >正确的回答加上错误的回答一共应该是每一个context 10条，不能多也不能少。
2. When torch.cuda.is_available() return true, still get error “RuntimeError: No CUDA GPUs are available”: (已解决）
    >把 os.environ["CUDA_VISIBLE_DEVICES"] = "1"，改成os.environ["CUDA_VISIBLE_DEVICES"] = "0"。 
     因为0是google colab上面的cuda device 的id。
3. masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), AttributeError: 'str' object has no attribute 'view'
    >重新安装transformers==2.8.0
4. ret = input.log_softmax(dim) RuntimeError: CUDA out of memory.
    >Decrease the batch size help
5. Input, output and indices must be on the current device.（已解决）
    >需要把tensor 传到cuda 上面， 即ids = to_cuda(ids)




Thought
===============
1. 英文数据过少的情况如何有效的提高模型的准确性？
2. 需要针对dog text 数据重新做post-training, 尝试使用其他的英文数据已经训练好的post-training 模型，
    效果一般。因此需要复现这篇论文中使用的post-training 方法。
3. 英文数据过少的情况如何有效的提高数据结构

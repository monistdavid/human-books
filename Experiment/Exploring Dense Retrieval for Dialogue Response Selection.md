Link    
===============
<p>

https://arxiv.org/pdf/2110.06612v1.pdf

https://github.com/gmftbyGMFTBY/SimpleReDial-v1.git


</p>


Question   
===============
1. 在训练补充内容的模型时，显示CUDA out of memory （已解决）
    >减少原本的batch-size 64 为 16，训练时间从30分钟变为2个小时，但是训练结果没有影响。


Thought
===============
1. 与 1/20/2022 提出思考类似，是否在狗狗的对话方面也能起到较好的效果，在如此大量的语句中进行分析，他的效率和准确率如何？
2. 如果使用的数据不规则，例如有的是对话句子，有的是背景描述，这个模型能否进行好的选择。
3. 是否因为数据太少导致它无法正确的找到input output 之间的关联性, 是否需要也做一个fine-tuning 在retrieval-base chatbot 上面？

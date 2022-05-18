Link    
===============
<p>

https://github.com/nawnoes/pytorch-meena

https://arxiv.org/pdf/2001.09977v3.pdf


</p>

Train   
===============


Question   
===============
1. ValueError: type of None unknown: <class 'NoneType'>:
    >在vocab里面加上[PAD][UNK][CLS][SEP][MASK]
2. pytorch/aten/src/ATen/native/cuda/Indexing.cu:699: indexSelectLargeIndex: block: [263,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
    >Use CPU instead to find out what the problem is
3. IndexError: index out of range in self 
    >1. Any input more than input_dim - 1 
    >2. Any input less than zero</p>
5. Runtime Error on tta “softmax_lastdim_kernel_impl” not implemented for ‘Half’
    >Turn CPU to GPU, it is related to FP16.
6. RuntimeError: Error(s) in loading state_dict for BertModel: Missing key(s) in state_dict: "embeddings.position_ids". 
    >Wrong version of transformer



Thought
===============
1. 是否因为数据太少导致它无法正确的找到input output 之间的关联性
是否需要也做一个fine-tuning 在retrieval-base chatbot 上面？

#Finetune blenderbot-400M-distill with dog dialogues (Huggingface) .

```
# Import the model class and the tokenizer

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration from torch.utils.data import Dataset

# Download and setup the model and tokenizer

tokenizer_open = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model_open = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill",
max_position_embeddings=1048, ignore_mismatched_sizes=True)
import torch

class Dataset(torch.utils.data.Dataset):
def __init__(self, inputs, outputs):
self.inputs = inputs self.outputs = outputs

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
        # outputs = tokenizer_open(self.output[idx], max_length=64)
        item['labels'] = torch.tensor(self.outputs['input_ids'][idx])
        return item
 
    def __len__(self):
        return len(self.outputs['input_ids'])

train_dataset = Dataset(train_input_embedding, train_output_embedding)
valid_dataset = Dataset(test_input_embedding, test_output_embedding)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
output_dir="./results", learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=8,
num_train_epochs=3, weight_decay=0.01,
)

trainer = Trainer(
model=model_open, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, tokenizer=tokenizer_open
)

trainer.train()
trainer.save_model("results")
```

#Finetune blenderbot-400M-distill with dog dialogues (Parlai) .

```
from parlai.scripts.display_data import DisplayData
from parlai.core.teachers import register_teacher, DialogTeacher

@register_teacher("dog")
class DogTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # opt is the command line arguments.
        
        # What is this shared thing?
        # We make many copies of a teacher, one-per-batchsize. Shared lets us store 
        
        # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
        # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
        # the fold name (train/valid/test) + ".txt"
        print(opt)
        opt['datafile'] = "/content/gdrive/MyDrive/Human_Books/doge/deploy_doge_2/dev.txt"
        self.id = "dog"
        super().__init__(opt, shared)
    
    def setup_data(self, datafile):
        # filename tells us where to load from.
        # We'll just use some hardcoded data, but show how you could read the filename here:
        print(f" ~~ Loading from {datafile} ~~ ")
        with open(datafile) as data_file:
          for line in data_file:
            content = ' '.join(line.split('\t')[1:-2])
            question = line.split('\t')[-2]
            answers = line.split('\t')[-1]
            yield {"text": content + "\n" + question, "labels": answers}, True
        
        # setup_data should yield tuples of ((text, label), new_episode)
        # That is ((str, str), bool)
        
        # first episode
        # notice how we have call, response, and then True? The True indicates this is a first message
        # in a conversation
        # yield ('Hello', 'Hi'), True
        # # Next we have the second turn. This time, the last element is False, indicating we're still going
        # yield ('How are you', 'I am fine'), False
        # yield ("Let's say goodbye", 'Goodbye!'), False
        
        # # second episode. We need to have True again!
        # yield ("Hey", "hi there"), True
        # yield ("Deja vu?", "Deja vu!"), False
        # yield ("Last chance", "This is it"), False
        
class DefaultTeacher(DogTeacher):
    pass
    
    
    
from parlai.scripts.train_model import TrainModel
!parlai train_model -t dog -m transformer/generator --multitask-weights 1,3,3,3 
--init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict 
--embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 
--learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu 
--fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 
--optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 
--update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 
-vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True 
--model-file /content/gdrive/MyDrive/Human_Books/doge/deploy_doge_2/open/blender/test_train_90M
```


1. If the finetune epoch sets too large and the accuracy getting too high, the model is overfitting.
2. If the finetune epoch sets too low and the accuracy getting not that high, the model generate sentence better.
   However, it produces strange mood of the dog personality.


#Finetune blenderbot-400M-distill with dog dialogues.

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

1. If the finetune epoch sets too large and the accuracy getting too high, the model is overfitting.
2. If the finetune epoch sets too low and the accuracy getting not that high, the model generate sentence better.
   However, it produces strange mood of the dog personality.


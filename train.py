import torch, transformers, pyreft 
import pandas as pd 
from colorama import init, Fore

init()

model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='cuda', 
    # cache_dir='./workspace', 
    token=''
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, model_max_tokens=2048, use_fast=False, 
    padding_side="right", token=''
)
tokenizer.pad_token = tokenizer.unk_token 

def prompt_template(prompt): 
    return f"""<s>[INST]<<sys>>You are a helpful assistant<</sys>>
        {prompt}
        [/INST]"""

# # Test case
# prompt = prompt_template("who is Welerson Luiz Maduro?") 
# # print(Fore.CYAN + prompt) 
# tokens = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
# response = model.generate(tokens)
# print(Fore.MAGENTA + tokenizer.decode(response[0])) 

# Get the reft model 
reft_config = pyreft.ReftConfig(
    representations={
        "layer":15,
        "component":"block_output", 
        "low_rank_dimension":4,
        "intervention":pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size, low_rank_dimension=4
        ) 
    }
)
 
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device('cuda')

# GRAB Data
df = pd.read_csv('knowledgeoverride-maduro.csv') 
X = df['Prompt'].values 
y = df['Response'].values 

# Operate on last token 
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, 
    model, 
    [prompt_template(x) for x in X], 
    y 
)

# # Create a custom ReftTrainer subclass that fixes the compute_loss issue
# class CustomReftTrainer(pyreft.ReftTrainerForCausalLM):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Override to ignore the num_items_in_batch parameter that's causing the error
#         if "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(**inputs)
#         if labels is not None:
#             # Add back labels for loss computation
#             loss = self.label_smoother(outputs, labels)
#         else:
#             # We don't use .loss here since the model may return tuples instead of ModelOutput
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
#         return (loss, outputs) if return_outputs else loss

class CustomReftTrainer(pyreft.ReftTrainerForCausalLM):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Simply ignore the num_items_in_batch argument
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


# Training arguments 
training_arguments = transformers.TrainingArguments(
    num_train_epochs=100, 
    output_dir='./models', 
    per_device_train_batch_size=2, 
    learning_rate=2e-3, 
    logging_steps=20
)

# Trainer for the reft model 
trainer = CustomReftTrainer(
# trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, 
    tokenizer=tokenizer, 
    args=training_arguments, 
    **data_module
)

# Train the model!!
_ = trainer.train() 

# Save the model 
reft_model.set_device('cpu') 
reft_model.save(
    save_directory='./trained_intervention'
)

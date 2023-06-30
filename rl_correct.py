# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
from typing import List, Dict

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from dataset_train_model import SummaryLMDataset
import pandas as pd
from datasets_bert import SBSSet_dict, TruthCheckSet_dict
import transformers
from torch.utils.data import DataLoader
from transformers import DataCollator, DataCollatorWithPadding
import functools
import random
from typing import Union

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
tqdm.pandas()

device = "cuda:0"

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        default="/home/upayuryeva/workfolder/titles/trl_bert_as_reward_model/gpt_checkpoint/rugpt3medium_sum_gazeta", 
        metadata={"help": "the model name"}
        )
    tokenizer_name: Optional[str] = field(
        default="/home/upayuryeva/workfolder/titles/trl_bert_as_reward_model/gpt_checkpoint/rugpt3medium_sum_gazeta", 
        metadata={"help": "the tokenizer name"}
        )
    reward_model_name: Optional[str] = field(
        default="/home/upayuryeva/workfolder/titles/bert_andrey/model/sbs", 
        metadata={"help": "the reward model name"}
        )
    # log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=2, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=10, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.1,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


# parser = HfArgumentParser(ScriptArguments)
# script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
script_args = ScriptArguments
reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    # log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# train_dataset = load_dataset("parquet", data_files="/home/upayuryeva/workfolder/titles/trl_bert_as_reward_model/data/toloka_sample_13k_top50gmv_07062023_inferenced.parquet")
# train_dataset = train_dataset.select(range(100000))
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
train_dataset = pd.read_parquet("/home/upayuryeva/workfolder/titles/trl_bert_as_reward_model/data/toloka_sample_13k_top50gmv_07062023_inferenced.parquet").iloc[:12992]
test_dataset = pd.read_parquet("/home/upayuryeva/workfolder/titles/trl_bert_as_reward_model/data/test_data.parquet")
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size":1,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    ds = dataset
    original_columns = ds.column_names["train"]
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for text in examples["title"]:
            tokenized_question = tokenizer(text, truncation=True)
            new_examples["query"].append(text)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds

class SummaryDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            original_records: List[Dict],
            max_source_char_len: int = 270,
            max_target_char_len: int = 128,
            max_source_tokens_count: int = 196,
            max_target_tokens_count: int = 128,
            source_field: str = "title",
            category_promt: str = "category_path",
            attrs_field: str = "extension",
            sep_phrase: str = "</sep>",
            conditional_gen_prompt: str = ""
    ):
        self.original_records = original_records
        self.max_source_char_len = max_source_char_len
        self.max_target_char_len = max_target_char_len
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.source_field = source_field
        self.category_promt = category_promt
        self.sep_phrase = sep_phrase
        self.attrs_field = attrs_field
        self.tokenizer = tokenizer
        self.records = []
        for record in original_records:
            tensors = self.convert_pair(
                category=record[category_promt],
                text=str(record[source_field]),
                attrs=record[attrs_field],
                conditional_gen_prompt=conditional_gen_prompt
            )
            self.records.append(tensors)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        return self.records[index]
    def convert_pair(self, category, text, attrs, conditional_gen_prompt):
        raise NotImplementedError


# uses all three tokens: pad and eos from tokenizer
class InputDS(SummaryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def convert_pair(
            self,
            category,
            text,
            attrs,
            conditional_gen_prompt
    ):
        if len(text + self.sep_phrase) >= self.max_source_char_len:
            text = text
        else:
            text = text + self.sep_phrase + attrs
        text = (
                category  # категория
                + self.sep_phrase
                + text[:self.max_source_char_len]  # Оригинальный заголовок:
                # + self.sep_phrase
        )
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True,
        )["input_ids"]

        input_ids = [self.tokenizer.bos_token_id] + input_ids

        unpadded_seq_len = len(input_ids)

        max_length = (
                self.max_source_tokens_count + 1  # <bos> + inp
        )

        padding = [
            self.tokenizer.pad_token_id for i in range(unpadded_seq_len, max_length)
        ]
        input_ids.extend(padding)


        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "query": text,
            "input_ids": input_ids,
            # "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
    

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = InputDS(original_records=train_dataset.to_dict("records"), tokenizer=tokenizer)
test_ds = InputDS(original_records=test_dataset.to_dict("records"), tokenizer=tokenizer)
# print(type(dataset))

# def collator(data):
#     return dict((key, [d[key] for d in data]) for key in data[0])
# collator = DataCollatorWithPadding(tokenizer, padding="longest", max_length=196)

def custom_collate_fn(batch):
    # Extract input_ids and attention_mask from each sample in the batch
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]

    # # Pad sequences
    # padding = tokenizer.pad_token_id
    # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding)
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Retrieve labels
    queries = [sample["query"] for sample in batch]

    # return {"input_ids": [row_tensor for row_tensor in input_ids], "attention_mask": [row_tensor for row_tensor in attention_mask], "query": queries}
    return {"input_ids": input_ids, "attention_mask": attention_mask, "query": queries}


def custom_collate_fn_test(batch):
    # Extract input_ids and attention_mask from each sample in the batch
    input_ids = [sample["input_ids"].to(device) for sample in batch]
    attention_mask = [sample["attention_mask"].to(device) for sample in batch]

    # # Pad sequences
    # padding = tokenizer.pad_token_id
    # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding)
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Retrieve labels
    queries = [sample["query"] for sample in batch]

    # return {"input_ids": [row_tensor for row_tensor in input_ids], "attention_mask": [row_tensor for row_tensor in attention_mask], "query": queries}
    return {"input_ids": input_ids, "attention_mask": attention_mask, "query": queries}


# Create a DataCollatorWithPadding instance using the custom collate function
collator = DataCollator(custom_collate_fn)

collator_test =  DataCollator(custom_collate_fn_test)

test_dataloader = DataLoader(
    test_ds, shuffle=False, collate_fn=collator_test, batch_size=config.batch_size,
)

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)


optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=model_ref,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model=reward_model_name,
#     device_map={"": current_device},
#     model_kwargs={"load_in_8bit": True},
#     tokenizer=tokenizer,
#     return_token_type_ids=False,
# )

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

reward_model_name_sbs = "/home/upayuryeva/workfolder/titles/bert_andrey/model/sbs"
reward_model_sbs = transformers.BertForSequenceClassification.from_pretrained(
    reward_model_name_sbs, num_labels=1
)
reward_model_sbs = reward_model_sbs.to(device)
reward_model_sbs = reward_model_sbs.eval()
reward_tokenizer_sbs = transformers.BertTokenizerFast.from_pretrained(reward_model_name_sbs, padding_side='left')

reward_model_name_truth = "/home/upayuryeva/workfolder/titles/bert_andrey/model/truth"
reward_model_truth = transformers.BertForSequenceClassification.from_pretrained(
    reward_model_name_truth, num_labels=2
)
reward_model_truth = reward_model_truth.to(device)
reward_model_truth = reward_model_truth.eval()
reward_tokenizer_truth = transformers.BertTokenizerFast.from_pretrained(reward_model_name_truth, padding_side='left')


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    print(f"TRAIN EPOCH {epoch}")
    question_tensors = batch["input_ids"]
    attention_mask_q = batch["attention_mask"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        # attention_mask=attention_mask_q,
        return_prompt=False,
        max_new_tokens = output_max_length,
        # length_sampler=output_length_sampler,
        # **generation_kwargs,
    )

    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True, clean_up_tokenization_spaces=True,)

    responses_clean = [response.split("</sep>")[-1] for response in responses]

    batch["response"] = responses_clean

    with torch.no_grad():

        reward_ds_sbs = SBSSet_dict(dataset=batch, title_in="query", title_out="response", tokenizer=reward_tokenizer_sbs, max_length=250)
        reward_ds_truth = TruthCheckSet_dict(dataset=batch, title_in="query", title_out="response", tokenizer=reward_tokenizer_truth, max_length=250)


        reward_collator_sbs = DataCollatorWithPadding(reward_tokenizer_sbs, padding="longest")
        reward_collator_truth = DataCollatorWithPadding(reward_tokenizer_truth, padding="longest")

        reward_dataloader_sbs = DataLoader(
            reward_ds_sbs, shuffle=False, collate_fn=reward_collator_sbs, batch_size=config.batch_size,
        )
        reward_dataloader_truth = DataLoader(
            reward_ds_sbs, shuffle=False, collate_fn=reward_collator_truth, batch_size=config.batch_size,
        )

        batch_sbs = next(iter(reward_dataloader_sbs))
        batch_truth = next(iter(reward_dataloader_truth))

        for key in batch_sbs.keys():
            if isinstance(batch_sbs[key], torch.Tensor):
                batch_sbs[key] = batch_sbs[key].to(device)

        for key in batch_truth.keys():
            if isinstance(batch_truth[key], torch.Tensor):
                batch_truth[key] = batch_truth[key].to(device)

        logits = reward_model_sbs(**batch_sbs).logits
        scores_batch_sbs = torch.sigmoid(logits).squeeze()

        logits = reward_model_truth(**batch_truth).logits
        scores_batch_thruth = torch.softmax(logits, dim=1)[:, 1]

        # print(scores_batch_sbs.shape)
        # print(scores_batch_thruth.shape)

        scores = scores_batch_sbs * scores_batch_thruth

        # print(scores_batch_sbs[0], scores_batch_thruth[0], scores[0])
        rewards = [raw_tensor for raw_tensor in scores]

        question_tensors = [raw_tensor for raw_tensor in question_tensors]
        response_tensors = [raw_tensor for raw_tensor in response_tensors]
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and (((epoch + 1) % script_args.save_freq == 0) or epoch == 0):

        ## generate and save model outputs

        print("START CHECK GENERATION")

        responses_all = []
        scores_sbs_all = []
        scores_truth_all = []
        
        for epoch_gen, batch_gen in tqdm(enumerate(test_dataloader)):
            print(f"TRAIN EPOCH {epoch} GENERATE EPOCH {epoch_gen}")

            question_tensors = batch_gen["input_ids"]

            response_tensors = ppo_trainer.generate(
                    question_tensors,
                    # attention_mask=attention_mask_q,
                    return_prompt=False,
                    max_new_tokens = output_max_length,
                    # length_sampler=output_length_sampler,
                    # **generation_kwargs,
                )

            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True, clean_up_tokenization_spaces=True,)

            responses_clean = [response.split("</sep>")[-1] for response in responses]

            batch_gen["response"] = responses_clean
            responses_all.extend(responses_clean)

            with torch.no_grad():

                reward_ds_sbs = SBSSet_dict(dataset=batch_gen, title_in="query", title_out="response", tokenizer=reward_tokenizer_sbs, max_length=250)
                reward_ds_truth = TruthCheckSet_dict(dataset=batch_gen, title_in="query", title_out="response", tokenizer=reward_tokenizer_truth, max_length=250)


                reward_collator_sbs = DataCollatorWithPadding(reward_tokenizer_sbs, padding="longest")
                reward_collator_truth = DataCollatorWithPadding(reward_tokenizer_truth, padding="longest")

                reward_dataloader_sbs = DataLoader(
                    reward_ds_sbs, shuffle=False, collate_fn=reward_collator_sbs, batch_size=config.batch_size,
                )
                reward_dataloader_truth = DataLoader(
                    reward_ds_sbs, shuffle=False, collate_fn=reward_collator_truth, batch_size=config.batch_size,
                )

                batch_sbs = next(iter(reward_dataloader_sbs))
                batch_truth = next(iter(reward_dataloader_truth))

                for key in batch_sbs.keys():
                    if isinstance(batch_sbs[key], torch.Tensor):
                        batch_sbs[key] = batch_sbs[key].to(device)

                for key in batch_truth.keys():
                    if isinstance(batch_truth[key], torch.Tensor):
                        batch_truth[key] = batch_truth[key].to(device)

                logits = reward_model_sbs(**batch_sbs).logits
                scores_batch_sbs = torch.sigmoid(logits).squeeze()

                logits = reward_model_truth(**batch_truth).logits
                scores_batch_thruth = torch.softmax(logits, dim=1)[:, 1]

                scores_batch_sbs = scores_batch_sbs.detach().cpu().numpy().tolist()
                scores_batch_thruth = scores_batch_thruth.detach().cpu().numpy().tolist()

                scores_sbs_all.extend(scores_batch_sbs)
                scores_truth_all.extend(scores_batch_thruth)

        test_dataset[f"rl_epoch{epoch}_responses"] = responses_all
        test_dataset[f"rl_epoch{epoch}_sbs_scores"] = scores_sbs_all
        test_dataset[f"rl_epoch{epoch}_truth_scores"] = scores_truth_all

        test_dataset.to_parquet("trl_responses.parquet")
        print("END CHECK GENERATION")

        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
            


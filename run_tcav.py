import argparse
from typing import Any
import jsonlines
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from captum.concept import TCAV, Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset


class LlamaCompletionConditionalLikelihood(nn.Module):
    
    def __init__(self, llama: LlamaForCausalLM, tokenizer: LlamaTokenizer):
        super().__init__()
        self.llama = llama
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        llama = LlamaForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        return cls(llama, tokenizer)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

    def prepare_encodings(self, prompt, completion):
        prompt_encodings = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        completion_encodings = self.tokenizer(completion, return_tensors='pt', add_special_tokens=False)

        input_ids = torch.cat([prompt_encodings.input_ids, completion_encodings.input_ids], dim=1)
        attention_mask = torch.cat([prompt_encodings.attention_mask, completion_encodings.attention_mask], dim=1)

        input_ids = torch.cat([
            self.tokenizer.bos_token_id * torch.ones_like(completion_encodings.input_ids[:,-1:]),
            input_ids,
            # self.tokenizer.eos_token_id * torch.ones_like(completion_encodings.input_ids[:,-1:])
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones_like(completion_encodings.attention_mask[:,-1:]),
            attention_mask,
            # torch.ones_like(completion_encodings.attention_mask[:,-1:])
        ], dim=1)

        completion_ids = input_ids.clone()
        completion_ids[:,:-completion_encodings.input_ids.size(1)] = -100

        return input_ids, attention_mask, completion_ids

    def forward(self, input_ids: torch.Tensor, attention_mask=None, completion_ids=None, reduction='sum'):
        """
        Passes a sequence a sequence to the model and computes the log-likelihood
        of the tokens with values given by completion_ids.
        
        This can be used to model the log-likelihood of a completion conditioned
        on a prompt by calling model(*model.prepare_encodings(prompt, completion)).
        
        Args:
            input_ids:
                tensor of shape (batch_size, seq_len)
                The input ids for the entire sequence
            attention_mask:
                tensor of shape (batch_size, seq_len)
                The attention mask corresponding to input_ids
            completion_ids:
                tensor of shape (batch_size, seq_len)
                The ides of the completion, with -100 values for the prompt,
                which are ignored in the loss calculation.
            reduction:
                str, either 'mean' or 'sum'
                How to reduce the loss. Defaults to 'mean'.
        """
        logits = self.llama(input_ids=input_ids, attention_mask=attention_mask).logits

        if completion_ids is None:
            completion_ids = input_ids.clone()

        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = completion_ids[:, 1:].contiguous().view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction=reduction)

        return -loss
    

class TCAV_LMCompletionPipeline:
    """
    Run TCAV of a concept on a language model completion
    """

    def __init__(
        self,
        model: LlamaCompletionConditionalLikelihood,
        device: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.tcav = TCAV(self.model, layers=["llama.model"])
    
    def assemble_concept(self, name, id, concepts_path):
        def _get_tensor_from_filename(filename):
            lines = Path(filename).read_text().strip().split("\n")
            tokenized = self.model.tokenizer(
                lines,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors='pt',
            )
            for row in tokenized.input_ids:
                yield row.to(self.device)

        dataset = CustomIterableDataset(_get_tensor_from_filename, concepts_path)
        concept_iter = dataset_to_dataloader(dataset, batch_size=1)
        return Concept(id=id, name=name, data_iter=concept_iter)

    def interpret(self, eval_prompt, eval_completion, experimental_sets):
        enc = self.model.prepare_encodings(eval_prompt, eval_completion)
        print("Encodings prepared")
        print(type(enc))
        return self.tcav.interpret(
            enc,
            experimental_sets=experimental_sets,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--eval_file", type=str, default="eval.jsonl")

    args = parser.parse_args()

    model = LlamaCompletionConditionalLikelihood.from_pretrained(args.model_id)
    tcav = TCAV_LMCompletionPipeline(model, 'cpu')
    
    western_names = tcav.assemble_concept('western-names', 0, "./names-western.csv")
    russian_names = tcav.assemble_concept('russian-names', 1, "./names-russian.csv")
    # general_names = tcav.assemble_concept('general-names', 2, "./names-general.csv")

    print("Concepts assembled")
    
    eval_prompts = []
    eval_completions = []
    with jsonlines.open(args.eval_file) as reader:
        for row in reader:
            eval_prompts.append(row['prompt'])
            eval_completions.append(row['completion'])

    print("Prompts and completions gathered")
    
    for prompt, completion in zip(eval_prompts, eval_completions):
        outputs = tcav.interpret(
            prompt,
            completion,
            experimental_sets=[
                [western_names, russian_names],
                # [russian_names, general_names],
            ],
        )
        print("Output interpreted")
        
        print()
        print(prompt, completion)
        print(outputs['0-1']['llama.model'])
        # print(outputs['0-2']['llama.model'])
        # print(outputs['1-2']['llama.model'])

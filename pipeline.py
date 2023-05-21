import re
import torch
import numpy as np
from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from utils.consts import END_KEY, PROMPT_FOR_GENERATION_FORMAT, RESPONSE_KEY

import logging

# Create and configure logger
logging.basicConfig(
    filename="newfile.log", format="%(asctime)s %(message)s", filemode="w"
)
# Create a custom logger
logger = logging.getLogger(__name__)
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def setup_model_for_generation(model_name, tokenizer_name, lora=False, lora_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    added_tokens = tokenizer.add_special_tokens(
        {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
    )

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    if lora:
        model = PeftModelForCausalLM.from_pretrained(
            model, lora_path, device_map="auto", torch_dtype=torch.float16
        )
        model.to(dtype=torch.float16)

    logger.info(
        f"Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB"
    )

    return model, tokenizer


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        ValueError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(
            f"Expected only a single token for '{key}' but found {token_ids}"
        )
    return token_ids[0]


class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self,
        *args,
        do_sample: bool = True,
        max_new_tokens: int = 256,
        top_p: float = 0.92,
        top_k: int = 0,
        **kwargs,
    ):
        super().__init__(
            *args,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    def _sanitize_parameters(self, return_instruction_text=False, **generate_kwargs):
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (
                token
                for token in self.tokenizer.additional_special_tokens
                if token.startswith(RESPONSE_KEY)
            ),
            None,
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(
                    self.tokenizer, tokenizer_response_key
                )
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id,
            "return_instruction_text": return_instruction_text,
        }

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )[0].cpu()
        instruction_text = model_inputs.pop("instruction_text")
        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "instruction_text": instruction_text,
        }

    def postprocess(
        self,
        model_outputs,
        response_key_token_id,
        end_key_token_id,
        return_instruction_text,
    ):
        sequence = model_outputs["generated_sequence"]
        instruction_text = model_outputs["instruction_text"]

        # The response will be set to this variable if we can identify it.
        decoded = None

        # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
        if response_key_token_id and end_key_token_id:
            # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
            # prompt, we should definitely find it.  We will return the tokens found after this token.
            response_pos = None
            response_positions = np.where(sequence == response_key_token_id)[0]
            if len(response_positions) == 0:
                logger.warn(
                    f"Could not find response key {response_key_token_id} in: {sequence}"
                )
            else:
                response_pos = response_positions[0]

            if response_pos:
                # Next find where "### End" is located.  The model has been trained to end its responses with this
                # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                # this token, as the response could be truncated.  If we don't find it then just return everything
                # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                end_pos = None
                end_positions = np.where(sequence == end_key_token_id)[0]
                if len(end_positions) > 0:
                    end_pos = end_positions[0]

                decoded = self.tokenizer.decode(
                    sequence[response_pos + 1 : end_pos]
                ).strip()
        else:
            # Otherwise we'll decode everything and use a regex to find the response and end.

            fully_decoded = self.tokenizer.decode(sequence)

            # The response appears after "### Response:".  The model has been trained to append "### End" at the
            # end.
            m = re.search(
                r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL
            )

            if m:
                decoded = m.group(1).strip()
            else:
                # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                # return everything after "### Response:".
                m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                if m:
                    decoded = m.group(1).strip()
                else:
                    logger.warn(f"Failed to find response in:\n{fully_decoded}")

        if return_instruction_text:
            return {"instruction_text": instruction_text, "generated_text": decoded}

        return {"generated_text": decoded}


def generate_response(
    instruction: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    return_instruction_text: bool = False,
    **kwargs,
) -> str:
    """Given an instruction, uses the model and tokenizer to generate a response.  This formats the instruction in
    the instruction format that the model was fine-tuned on.

    Args:
        instruction (str): _description_
        model (PreTrainedModel): the model to use
        tokenizer (PreTrainedTokenizer): the tokenizer to use

    Returns:
        str: response
    """

    generation_pipeline = InstructionTextGenerationPipeline(
        model=model, tokenizer=tokenizer, **kwargs
    )
    return generation_pipeline(
        instruction, return_instruction_text=return_instruction_text
    )


def generate(prompt, model, tokenizer, **generate_kwargs):
    prompt_encodings = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_encodings.input_ids.to(model.device)
    attention_mask = prompt_encodings.attention_mask.to(model.device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded[len(prompt) :]

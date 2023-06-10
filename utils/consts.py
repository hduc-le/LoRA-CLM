DEFAULT_SEED = 42
DEFAULT_INPUT_MODEL = "EleutherAI/gpt-j-6B"


INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that does not contain an input string.  The instruction by itself has enough information
# to respond.  For example, the instruction might ask for the year a historic figure was born.
INTRO_NO_INPUT_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_NO_INPUT_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)
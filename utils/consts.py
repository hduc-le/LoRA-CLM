DEFAULT_SEED = 42
DEFAULT_INPUT_MODEL = "databricks/dolly-v2-3b"

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
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

import logging
from colorlog import ColoredFormatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    #    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green,bold",
        "INFOV": "cyan,bold",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)
ch.setFormatter(formatter)

LOG = logging.getLogger("rn")
LOG.setLevel(logging.DEBUG)
LOG.handlers = []  # No duplicated handlers
LOG.propagate = False  # workaround for duplicated logs in ipython
LOG.addHandler(ch)
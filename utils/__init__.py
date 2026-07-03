import json
import os

import openai

from .featurize import *
from .n_gram import *

try:
    from .write_logprobs import *
except Exception:
    # Some workflows only need the lightweight helpers and should not fail
    # because the gated Llama tokenizer is unavailable.
    pass

openai_path = ""
if os.path.exists("../../openai.config"):
    openai_path = "../../openai.config"
elif os.path.exists("../openai.config"):
    openai_path = "../openai.config"
elif os.path.exists("openai.config"):
    openai_path = "openai.config"

if openai_path:
    openai_config = json.loads(open(openai_path).read())
    openai.api_key = openai_config["api_key"]
    openai.organization = openai_config["organization"]

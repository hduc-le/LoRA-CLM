import torch
from argparse import ArgumentParser
from utils.read import read_config
from termcolor import colored
from utils.consts import RESPONSE_KEY_NL
from pipeline import generate_response, setup_model_for_generation

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = setup_model_for_generation(**config["model_config"])
    model.eval(); model.to(device)
    
    response = generate_response(model=model, 
                                 tokenizer=tokenizer,
                                 instruction=args.prompt + " " + RESPONSE_KEY_NL,
                                 **config["generate_config"])
    
    if config["generate_config"]["return_instruction_text"]:
        print(colored("Instruction Text:\n", "light_green", attrs=["bold"]), response["instruction_text"])

    print(colored("Generated Text:\n", "light_green", attrs=["bold"]), response["generated_text"])
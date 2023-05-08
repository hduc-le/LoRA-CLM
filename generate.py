import torch
from argparse import ArgumentParser
from utils.read import read_config
from functional.pipeline import generate_response, setup_model_for_generation

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = setup_model_for_generation(**config["model_config"])
    model.eval()
    model.to(device)
    
    response = generate_response(instructino=args.prompt, 
                                 model=model, tokenizer=tokenizer, **config["generate_config"])
    
    print(response)
    

    

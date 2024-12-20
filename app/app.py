# %%
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import setup_model_for_generation, generate_response
from utils.load import read_config
from utils.consts import RESPONSE_KEY
from flask import Flask, render_template, request, redirect, jsonify
 
app = Flask(__name__, template_folder='template', static_folder='static')

######################## MODEL PREPARATION ########################

config = read_config("configs/generate.yaml")
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = setup_model_for_generation(**config["model_config"])
model.eval(); model.to(device)

###################################################################

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect("/")
    return render_template("index.html")
 
@app.route("/get")
def get_response():
    """Model inference and get response

    Returns:
        str: response text
    Ex: 
        >> response = model.generate(userText, **kwargs)
    """
    user_text = request.args.get("msg-text")
    model_response = generate_response(model=model,
                                 tokenizer=tokenizer,
                                 instruction=user_text+" "+RESPONSE_KEY,
                                 **config["generate_config"])

    return jsonify({"msg": (model_response["generated_text"]
                            .replace(RESPONSE_KEY, ""))})

if __name__ == "__main__":
    app.run(debug=True)
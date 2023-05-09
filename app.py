import torch
from functional.pipeline import setup_model_for_generation, generate_response
from utils.read import read_config
from utils.consts import LOG, RESPONSE_KEY
from flask import Flask, render_template, request, redirect
 
app = Flask(__name__, template_folder='template', static_folder='static')

######################## MODEL PREPARATION ########################
config = read_config("configs/generate.yaml")
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
    userText = request.args.get('msg')
    response = generate_response(model=model,
                                 tokenizer=tokenizer,
                                 instruction=userText+" "+RESPONSE_KEY,
                                 **config["generate_config"])
    return str(response["generated_text"])
 
if __name__ == "__main__":
    app.run(debug=True)
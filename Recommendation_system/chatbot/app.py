from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")  # Corrected the form data retrieval
    response = get_Chat_response(msg)
    return response

def get_Chat_response(text):
    chat_history_ids = None  # Initialize chat history
    for step in range(5):
        # Encode user input and add EOS token
        new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

        # Append new input tokens to chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate response
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response  # Return the first response

if __name__ == '__main__':
    app.run(debug=True)  # Added debug mode for error tracking

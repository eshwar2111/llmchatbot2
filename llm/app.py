from flask import Flask, request, jsonify
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    message = request.json["message"]
    input_ids = tokenizer.encode(message, return_tensors="tf")
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output_ids[0])

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
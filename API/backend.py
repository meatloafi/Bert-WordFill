from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn.functional import softmax

app = Flask(__name__, static_url_path='')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Print the original text input
        print("Original text:", text)

        # Tokenize the text
        tokenized_text = tokenizer.tokenize(text)
        # Print the tokenized text
        print("Tokenized text:", tokenized_text)

        try:
            masked_index = tokenized_text.index('[MASK]')
            print("Index of [MASK]:", masked_index)
        except ValueError:
            # If [MASK] is not found in the tokenized text, return an error message
            return jsonify({'error': '[MASK] token not found in the text.'})

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        # Print the tokens tensor
        print("Tokens tensor:", tokens_tensor)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs.logits

        # Print the predictions for the masked index
        print("Predictions at masked index:", predictions[0, masked_index])

        probabilities = softmax(predictions[0, masked_index], dim=0)

        # Get the top 5 most likely token IDs
        top5_indices = torch.topk(probabilities, 5).indices
        top5_tokens = tokenizer.convert_ids_to_tokens(top5_indices.tolist())

        # Print the top 5 most likely tokens
        print("Top 5 predicted tokens:", top5_tokens)

        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        # Print the predicted token
        print("Predicted token:", predicted_token)

        response = {
            'original_text': text,
            'filled_text': text.replace('[MASK]', predicted_token),
            'predicted_word': predicted_token,
            'top_predictions': top5_tokens
        }

        return render_template('result.html', response=response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)




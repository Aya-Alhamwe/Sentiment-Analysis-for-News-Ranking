from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize FastAPI
app = FastAPI()

# Load the trained Keras model and Tokenizer
model = tf.keras.models.load_model('my_model.h5')  # Replace with the correct path to the model
tokenizer = Tokenizer()  # Make sure to load or retrain the Tokenizer as needed

# Define the input model for the request
class Review(BaseModel):
    text: str

def clean_text_with_stopwords(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = text.strip()
    return text

# Define a function to predict the sentiment
def predict_sentiment(text: str):
    # Preprocess the input text
    cleaned_text = clean_text_with_stopwords(text)
    
    # Tokenize and pad the text (make sure to use the same tokenizer used during training)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=100)  # Ensure `maxlen` is the same as during training
    
    # Get predictions for each sentiment class (positive, negative, neutral)
    pos_pred, neg_pred, neu_pred = model.predict(padded_seq)
    
    # Return the highest predicted sentiment
    sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}
    sentiment = np.argmax([pos_pred, neg_pred, neu_pred])
    
    return sentiment_map[sentiment]

# Define the root endpoint to return HTML for the user interface
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>News Sentiment Analysis</title>
        <style>
            body { 
                font-family: 'Arial', sans-serif; 
                background: #212121; /* Dark background */
                margin: 0; 
                padding: 20px; 
                color: #e0e0e0; /* Light gray text for readability */
            }
            h1 { 
                text-align: center; 
                margin-bottom: 20px; 
                color: #64b5f6; /* Soft light blue */
                font-size: 2.5em; 
                text-shadow: 0 4px 10px rgba(100, 181, 246, 0.4); 
            }
            .container { 
                max-width: 600px; 
                margin: auto; 
                background: #333333; /* Dark container background */
                padding: 30px; 
                border-radius: 12px; 
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6); 
            }
            textarea { 
                width: 100%; 
                height: 150px; 
                border-radius: 8px; 
                padding: 12px; 
                background: #444444; /* Dark gray text area */
                color: #f1f1f1; /* Light text */
                font-size: 1.1em; 
                border: 1px solid #666666; /* Dark border */
            }
            button { 
                padding: 12px 18px; 
                background: #1e88e5; /* Dark blue background */
                color: white; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 1.1em; 
                font-weight: bold; 
                transition: background-color 0.3s ease; 
            }
            button:hover {
                background-color: #1565c0; /* Darker blue on hover */
            }
            .result { 
                margin-top: 20px; 
                padding: 20px; 
                background: #424242; /* Dark gray background for result */
                border-radius: 8px; 
                border-left: 4px solid #64b5f6; /* Soft blue border */
                color: #80deea; /* Light blue text */
                font-size: 1.1em; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>News Sentiment Analysis</h1>
            <textarea id="newsText" placeholder="Enter news text here..."></textarea>
            <button onclick="submitText()">Classify Sentiment</button>
            <div class="result" id="result"></div>
        </div>
        <script>
            async function submitText() {
                const text = document.getElementById('newsText').value;
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();
                document.getElementById('result').innerHTML = "Sentiment: " + data.sentiment;
            }
        </script>
    </body>
    </html>
    """

# Define the endpoint for sentiment prediction
@app.post("/predict")
async def predict(review: Review):
    sentiment = predict_sentiment(review.text)
    return {"sentiment": sentiment}

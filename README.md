# 📊 Sentiment Analysis & Ranking for Financial News  

## 🚀 Overview  
This project aims to analyze **financial news related to stocks** to assist investors in **improving decision-making**. The system does not only classify news into **positive, negative, or neutral**, but it also includes a **Ranking System** that evaluates the impact of each news article on the market, allowing for the identification of **the best investment opportunities available at the moment**.  


## 🔥 Key Features  
✅ **Financial news classification** into positive, negative, or neutral 📌  
✅ **Ranking System** to determine the impact of news on the stock market 📈  
✅ **Real-time news analysis** and recommendations for **top investment stocks** 💹  
✅ **Fully automated pipeline** from data collection to recommendation extraction ⚙️  
✅ **Implementation of MLOps practices** for model deployment and continuous improvement 🛠️  

---

## 🛠️ Tech Stack  
- **FastAPI** – for building the REST API
- **Git LFS** – for handling large files like datasets and models  
- **TensorFlow/Keras** – for training the sentiment analysis model  
- **Docker** – for containerizing the application  
- **MLOps** – for automating model deployment and versioning  

---

## 📦 Installation & Setup  

### 🔹 Clone the repository  
```bash
git clone https://github.com/Aya-Alhamwe/Sentiment-Analysis-for-News-Ranking.git
cd Sentiment-Analysis-for-News-Ranking
```

### 🔹 Install dependencies  
```bash
pip install -r requirements.txt
```

### 🔹 Run the FastAPI server  
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Now, the API is running on **http://localhost:8000** 🚀  

---

## 🧠 Model Details

The sentiment analysis model is built using **TensorFlow/Keras** and trained on financial news data. The ranking system evaluates each news item based on its potential market impact.

### 🔹 Sentiment Classification

The model uses the following architecture:

- **Preprocessing**: Tokenizing the text and converting it into padded sequences to ensure uniform input size.
- **Word2Vec Embeddings**: Pretrained embeddings are used for better representation of words in the corpus, capturing semantic meanings.
- **Bidirectional GRU Layers**: The **Bidirectional GRU** layers process the sequences in both forward and backward directions, capturing dependencies in the text for more accurate sentiment analysis.
- **Attention Layer**: This layer helps the model focus on important words, enhancing the understanding of sentiment and improving the classification accuracy.
- **Dense Layers**: These layers are used for the final classification of sentiments into **positive**, **negative**, or **neutral**.

### 🔹 Sentiment Classification  
- **Positive** 💚 – News that indicates a positive impact on stock prices  
- **Negative** 🔴 – News that suggests a negative impact on stock prices  
- **Neutral** ⚪ – News with little to no effect on stock prices  

### 🔹 News Ranking System  
The ranking model assigns an **importance score** to each news article based on:  
- **Market reactions to similar news in the past**  
- **The credibility of the news source**  
- **Keywords and sentiment intensity**  

---

## 🔄 MLOps Pipeline  

The project follows **MLOps best practices** for continuous model improvement:  
1. **Data Ingestion** – Collect and clean real-time financial news 📰  
2. **Preprocessing** – Remove noise, tokenize, and vectorize text 🔍  
3. **Model Training** – Fine-tune sentiment analysis & ranking models ⚡  
4. **Deployment** – Containerized with Docker & automated with CI/CD 🚀  
5. **Monitoring & Retraining** – Improve model based on new data 📊  

---

## 🐳 Running with Docker  
To run the application in a **Docker container**, use:  
```bash
docker build -t news-sentiment .
docker run -p 8000:8000 news-sentiment
```
The API will be available at **http://localhost:8000**.  

---

🌐 Deployment on Docker Hub

I have deployed the model on Docker Hub for easy access and distribution. You can pull the Docker image from the following link:

Docker Hub Repository: ayaalhamwe/my_news_model
To pull and run the Docker container, use:

bash
Copy
Edit
docker pull ayaalhamwe/my_news_model:latest
docker run -p 8000:8000 ayaalhamwe/my_news_model:latest
The API will be available at http://localhost:8000.

## 📡 API Endpoints  

### 🔹 Predict Sentiment  
**Endpoint:** `/predict`  
**Method:** `POST`  
**Request Body:**  
```json
{
  "text": "Stock prices surge after major investment announcement."
}
```
**Response:**  
```json
{
  "sentiment": "positive"
}
```

### 🔹 Get News Ranking  
**Endpoint:** `/rank-news`  
**Method:** `POST`  
**Request Body:**  
```json
{
  "news": [
    {"text": "Tech stocks are expected to rise due to new policies."},
    {"text": "Economic slowdown predicted, affecting major markets."}
  ]
}
```
**Response:**  
```json
{
  "ranked_news": [
    {"text": "Tech stocks are expected to rise due to new policies.", "score": 0.9},
    {"text": "Economic slowdown predicted, affecting major markets.", "score": 0.4}
  ]
}
```

---

## 🏗️ Future Improvements  
- 🔄 **Integrate real-time stock market data** to enhance ranking accuracy  
- 📈 **Improve ranking model** using reinforcement learning  
- ☁ **Deploy on cloud services** for scalability  

---

## 📜 License  
This project is **open-source** and available under the **MIT License**.  

---

🚀 **Developed by [Aya Alhamwe](https://github.com/Aya-Alhamwe) – AI & MLOps Enthusiast** 💡  
```

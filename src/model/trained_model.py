# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from keras.layers import Input, Embedding, Conv1D, Dropout, GlobalMaxPooling1D, Bidirectional, GRU, LayerNormalization, Dense, Attention
from keras.models import Model

# Function to split data
def split_data(news_df, sentiment_data):
    X_train, X_temp, y_train, y_temp = train_test_split(news_df, sentiment_data, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to tokenize and pad sequences
def tokenize_and_pad_sequences(X_train, X_val, tokenizer, max_length=100):
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_length)
    return X_train_pad, X_val_pad

# Function to create embedding matrix using Word2Vec
def create_embedding_matrix(news_df, tokenizer, embedding_dim=50):
    sentences = [sentence.split() for sentence in news_df['cleaned_text']]
    word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    
    return embedding_matrix

# Function to scale the sentiment data
def scale_sentiment_data(y_train, y_val, y_test):
    scaler = MinMaxScaler()

    # Extract individual columns of sentiment
    train_positive, train_negative, train_neutral = y_train["pos"], y_train["neg"], y_train["neu"]
    val_positive, val_negative, val_neutral = y_val["pos"], y_val["neg"], y_val["neu"]
    test_positive, test_negative, test_neutral = y_test["pos"], y_test["neg"], y_test["neu"]

    # Fit the scaler on all training labels
    train_labels = np.concatenate([train_negative, train_neutral, train_positive]).reshape(-1, 1)
    scaler.fit(train_labels)

    # Scale each sentiment score using the same scaler
    train_neg_scaled = scaler.transform(train_negative.values.reshape(-1, 1))
    val_neg_scaled = scaler.transform(val_negative.values.reshape(-1, 1))
    test_neg_scaled = scaler.transform(test_negative.values.reshape(-1, 1))

    train_neu_scaled = scaler.transform(train_neutral.values.reshape(-1, 1))
    val_neu_scaled = scaler.transform(val_neutral.values.reshape(-1, 1))
    test_neu_scaled = scaler.transform(test_neutral.values.reshape(-1, 1))

    train_pos_scaled = scaler.transform(train_positive.values.reshape(-1, 1))
    val_pos_scaled = scaler.transform(val_positive.values.reshape(-1, 1))
    test_pos_scaled = scaler.transform(test_positive.values.reshape(-1, 1))

    return train_pos_scaled, val_pos_scaled, test_pos_scaled, train_neg_scaled, val_neg_scaled, test_neg_scaled, train_neu_scaled, val_neu_scaled, test_neu_scaled

# Function to build the model
def build_model(embedding_matrix, max_length=100, embedding_dim=50):
    input_layer = Input(shape=(max_length,), name="input_data")
    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)(input_layer)

    conv_layer = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(embedding_layer)
    conv_layer = Dropout(0.4)(conv_layer)
    pooling_layer = GlobalMaxPooling1D()(conv_layer)

    gru_layer_1 = Bidirectional(GRU(256, return_sequences=True))(embedding_layer)
    gru_layer_1 = Dropout(0.4)(gru_layer_1)
    norm_1 = LayerNormalization()(gru_layer_1)

    attention_layer = Attention()([norm_1, norm_1, norm_1])
    attention_layer = Dropout(0.5)(attention_layer)

    gru_layer_2 = Bidirectional(GRU(128, return_sequences=False))(attention_layer)
    gru_layer_2 = Dropout(0.5)(gru_layer_2)
    norm_2 = LayerNormalization()(gru_layer_2)

    dense_shared = Dense(128, activation='relu')(norm_2)
    dense_shared = Dropout(0.5)(dense_shared)
    norm_shared = LayerNormalization()(dense_shared)

    # Multi-output
    pos_output = Dense(1, activation='sigmoid', name='positive_output')(Dense(64, activation='relu')(norm_shared))
    neg_output = Dense(1, activation='sigmoid', name='negative_output')(Dense(64, activation='relu')(norm_shared))
    neu_output = Dense(1, activation='sigmoid', name='neutral_output')(Dense(64, activation='relu')(norm_shared))

    model = Model(inputs=input_layer, outputs=[pos_output, neg_output, neu_output])
    return model

# Function to compile and train the model
def compile_and_train_model(model, X_train_pad, y_train, X_val_pad, y_val, epochs=5, batch_size=64):
    model.compile(optimizer='adam',
                  loss={'positive_output': 'binary_crossentropy', 
                        'negative_output': 'binary_crossentropy',
                        'neutral_output': 'binary_crossentropy'},
                  metrics={'positive_output': ['accuracy'], 
                           'negative_output': ['accuracy'], 
                           'neutral_output': ['accuracy']})

    history = model.fit(X_train_pad, 
                        {'positive_output': y_train[0], 
                         'negative_output': y_train[1], 
                         'neutral_output': y_train[2]},
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val_pad, 
                                         {'positive_output': y_val[0], 
                                          'negative_output': y_val[1], 
                                          'neutral_output': y_val[2]}))
    return history

# Example usage of the functions
X_train, X_val, X_test, y_train, y_val, y_test = split_data(news_df, sentiment_data)
X_train_pad, X_val_pad = tokenize_and_pad_sequences(X_train, X_val, tokenizer)
embedding_matrix = create_embedding_matrix(news_df, tokenizer)
train_pos_scaled, val_pos_scaled, test_pos_scaled, train_neg_scaled, val_neg_scaled, test_neg_scaled, train_neu_scaled, val_neu_scaled, test_neu_scaled = scale_sentiment_data(y_train, y_val, y_test)
model = build_model(embedding_matrix)
history = compile_and_train_model(model, X_train_pad, [train_pos_scaled, train_neg_scaled, train_neu_scaled], X_val_pad, [val_pos_scaled, val_neg_scaled, val_neu_scaled])

# Print training history
print("Training History:", history.history)

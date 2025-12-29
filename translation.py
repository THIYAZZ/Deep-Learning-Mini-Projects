import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense


english_sentences = [
    "hello", "how are you", "good morning", "thank you", "i love you",
    "what is your name", "see you later", "good night", "i am fine", "where is the station"
]

french_sentences = [
    "bonjour", "comment ça va", "bonjour", "merci", "je t'aime",
    "comment tu t'appelles", "à plus tard", "bonne nuit", "je vais bien", "où est la gare"
]

eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
eng_seq = eng_tokenizer.texts_to_sequences(english_sentences)
eng_seq = pad_sequences(eng_seq, padding='post')
eng_max_len = eng_seq.shape[1]

fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(french_sentences)
fr_seq = fr_tokenizer.texts_to_sequences(french_sentences)
fr_seq = pad_sequences(fr_seq, padding='post')
fr_max_len = fr_seq.shape[1]

X_train, X_test, y_train, y_test = train_test_split(eng_seq, fr_seq, test_size=0.2, random_state=42)

latent_dim = 64  

encoder_inputs = Input(shape=(eng_max_len,))
encoder_emb = Embedding(input_dim=len(eng_tokenizer.word_index)+1, output_dim=latent_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(fr_max_len,))
decoder_emb = Embedding(input_dim=len(fr_tokenizer.word_index)+1, output_dim=latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True)(decoder_emb, initial_state=encoder_states)
decoder_outputs = Dense(len(fr_tokenizer.word_index)+1, activation='softmax')(decoder_lstm)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


decoder_target_data = np.expand_dims(y_train, -1)
model.fit([X_train, y_train], decoder_target_data, batch_size=2, epochs=50, validation_split=0.2)

decoder_target_test = np.expand_dims(y_test, -1)
model.evaluate([X_test, y_test], decoder_target_test)

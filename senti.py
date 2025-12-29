import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras


docs = [
    # Positive examples
    ('good movie', 1),
    ('great film', 1),
    ('awesome show', 1),
    ('i love it', 1),
    ('fantastic', 1),
    ('really enjoyed the movie', 1),
    ('it was a wonderful experience', 1),
    ('amazing story and acting', 1),
    ('superb direction', 1),
    ('highly recommend it', 1),
    ('beautiful and touching', 1),
    ('excellent performance', 1),
    ('this film made me happy', 1),
    ('great visuals and music', 1),
    ('i liked every moment', 1),
    ('the plot was engaging', 1),
    ('very satisfying ending', 1),
    ('inspiring and emotional', 1),
    ('well made and entertaining', 1),
    ('masterpiece', 1),

    # Negative examples
    ('bad movie', 0),
    ('terrible film', 0),
    ('horrible show', 0),
    ('i hate it', 0),
    ('awful', 0),
    ('boring and slow', 0),
    ('worst experience ever', 0),
    ('poor acting and story', 0),
    ('disappointing movie', 0),
    ('i did not like it', 0),
    ('waste of time', 0),
    ('the plot was confusing', 0),
    ('bad direction and script', 0),
    ('not worth watching', 0),
    ('terrible visuals', 0),
    ('very disappointing ending', 0),
    ('dull and predictable', 0),
    ('hated every second', 0),
    ('no emotions at all', 0),
    ('complete disaster', 0)
]

texts = [d[0] for d in docs]
labels = np.array([d[1] for d in docs])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)


vocab_size = len(tokenizer.word_index) + 1
print(f'Vocabulary Size: {vocab_size}')
print('Word Index:', tokenizer.word_index)


encoded_texts = tokenizer.texts_to_sequences(texts)
print(f'\nEncoded Texts: {encoded_texts}')


max_length = max(len(e) for e in encoded_texts)
padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')
print(f'Padded Texts:\n{padded_texts}')

embedding_dim = 8

model = Sequential()

model.add(Embedding(input_dim=vocab_size,         
                    output_dim=embedding_dim,     
                    input_length=max_length))   
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


print("\n--- Training Model ---")
history = model.fit(padded_texts, labels, epochs=100, verbose=2)


print("\n--- Model Testing ---")
try:
    while True:
        in_text = input("\nEnter a sentence to test : ")
        if in_text.lower() == 'exit':
            break

        if not in_text.strip():
            continue

       
        encoded_test = tokenizer.texts_to_sequences([in_text])
        padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

        # Make a prediction
        prediction = model.predict(padded_test, verbose=0)[0][0]
        
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        print(f"Text: '{in_text}'")
        print(f"Prediction Score: {prediction:.4f}")
        print(f"Sentiment: {sentiment}")

except KeyboardInterrupt:
    print("\nExiting.")

import nltk, numpy as np
from nltk.corpus import treebank
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


nltk.download('treebank')
sents = treebank.tagged_sents()


words = sorted({w for s in sents for w, _ in s})
tags = sorted({t for s in sents for _, t in s})
w2i = {w: i+1 for i, w in enumerate(words)}
t2i = {t: i+1 for i, t in enumerate(tags)}
vocab_size, tag_size = len(w2i)+1, len(t2i)+1

X = [[w2i[w] for w, _ in s] for s in sents]
Y = [[t2i[t] for _, t in s] for s in sents]
maxlen = max(len(x) for x in X)
X = pad_sequences(X, maxlen, padding='post')
Y = pad_sequences(Y, maxlen, padding='post')
Y = np.array([to_categorical(y, num_classes=tag_size) for y in Y])


Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1)

inp = Input(shape=(maxlen,))
emb = Embedding(vocab_size, 128)(inp)
rnn_out = SimpleRNN(128, return_sequences=True)(emb)
out = Dense(tag_size, activation='softmax')(rnn_out)


model = Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(Xtr, Ytr, batch_size=32, epochs=3, validation_split=0.1, verbose=1)

print("\nTest Accuracy:", model.evaluate(Xte, Yte, verbose=0)[1])

test = ["rajesh", "dog", "palace"]
seq = pad_sequences([[w2i.get(w, 0) for w in test]], maxlen, padding='post')
pred = model.predict(seq)[0]
i2t = {i: t for t, i in t2i.items()}

tags_pred = [i2t.get(int(np.argmax(p)), 'UNK') for p in pred[:len(test)]]

print("\nSentence:", test)
print("Predicted POS tags:", tags_pred)

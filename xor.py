import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


input_size = 2
hidden_size = 2 

output_size = 1
epochs = 5000    

model = Sequential([
    Dense(hidden_size, input_dim=input_size, activation='relu'),
    Dense(output_size, activation='sigmoid')
])
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


print(f"Starting training with Keras ({epochs} epochs)...")

history = model.fit(X, y, epochs=epochs, verbose=0)
print("Training complete.")


print("\n--- Model's Learned Input-Output Sequence ---")
print(" Input  | Correct | Model's")
print("        | Answer  | Prediction (raw) | Prediction (clean)")
print("---------------------------------------------------------------")



predictions_raw = model.predict(X, verbose=0)

for i in range(len(X)):
    input_pair = X[i]
    correct_answer = y[i][0]
    predicted_raw = predictions_raw[i][0]
    predicted_clean = (predicted_raw > 0.5).astype(int) 
    
    inp_str = f"[{input_pair[0]}, {input_pair[1]}]"
    
    print(f" {inp_str} |    {correct_answer}    |   {predicted_raw:.4f} \t | \t {predicted_clean}")


loss_history = history.history['loss']

plt.plot(range(epochs), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (Binary Crossentropy)")
plt.title("Training Loss Over Epochs (Keras Model)")
plt.show()
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense

x=np.array ([1,2,3,4,5],dtype=float)
y=np.array ([1,4,9,16,25],dtype=float)
x=x.reshape(5,1,1)

model=Sequential([SimpleRNN(10,activation='relu',input_shape=(1,1)),Dense(1)])
model.compile(optimizer='adam',loss='mae')
model.fit(x,x,epochs=800,verbose=0)
print(model.predict(np.array([[6]])))
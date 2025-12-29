import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

x=np.array([[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]]])
y=([[4],[5],[6],[7]])


model=Sequential([LSTM(50,activation='tanh',input_shape=(3,1)),Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.fit(x,x,epochs=50,verbose=0)
print(model.predict(np.array([[[5],[6],[7]]])))
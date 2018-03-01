from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

class GeneratorNN:
    def __init__(self, hidden_size):
        self.init_epochs = 0
        self.nn = Sequential()
        self.nn.add(Dense(hidden_size, input_dim=1, activation="softplus"))
        self.nn.add(Dense(hidden_size))
        self.nn.add(Dense(hidden_size * 2, activation="tanh"))
        self.nn.add(Dense(hidden_size * 2, activation="tanh"))
        self.nn.add(Dense(hidden_size * 2, activation="tanh"))
        self.nn.add(Dense(1, activation="sigmoid"))
        sgd = SGD(lr=0.005, decay=0.95)
        self.nn.compile(optimizer=sgd, loss="mean_squared_error")

    def train(self, x, y, batch_size, epochs):
        self.nn.fit(x=x, y=y, batch_size=batch_size, epochs=epochs+self.init_epochs, initial_epoch=self.init_epochs)
        self.init_epochs += epochs

    def generate(self, x):
        pass

    def load(self):
        pass

    def save(self):
        pass
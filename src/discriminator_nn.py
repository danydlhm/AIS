from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD

class DiscriminatorNN:
    def __init__(self, hidden_size):
        self.init_epochs = 0
        self.nn = Sequential()
        self.nn.add(Dense(hidden_size * 2, activation="tanh", input_dim=1))
        self.nn.add(Dense(hidden_size * 2, activation="tanh"))
        self.nn.add(Dense(hidden_size * 2, activation="tanh"))
        self.nn.add(Dense(hidden_size * 2, activation="tanh"))
        self.nn.add(Dense(1, activation="sigmoid"))
        sgd = SGD(lr=0.005, decay=0.95)
        self.nn.compile(optimizer=sgd, loss="mean_squared_error")

    def train(self, x, y, batch_size, epochs):
        self.nn.fit(x=x, y=y, batch_size=batch_size, epochs=epochs+self.init_epochs, initial_epoch=self.init_epochs)
        self.init_epochs += epochs

    def evaluate(self):
        pass

    def load(self):
        pass

    def save(self):
        pass
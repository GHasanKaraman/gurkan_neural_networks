import numpy as np

class Layer:
    def __init__(self):
        self.layer_name = None
    def __repr__(self):
        return self.layer_name

class Dense(Layer):
    def __init__(self, units, activation = None, use_bias = True, **kwargs):
        self.layer_name = "Dense"
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        for key, value in kwargs.items():
            if key == "input_dim":
                self.input_dim = value

class Input(Layer):
    def __init__(self, input_dim):
        self.input_dim = input_dim

class Activation(Layer):
    pass

class ReLu(Activation, Layer):
    def __init__(self):
        self.layer_name = "ReLu"

class Sigmoid(Activation, Layer):
    def __init__(self):
        self.layer_name = "Sigmoid"

class Tanh(Activation, Layer):
    def __init__(self):
        self.layer_name = "Tanh"

class LeakyReLu(Activation, Layer):
    def __init__(self):
        self.layer_name = "LeakyReLu"

class Sequential:
    def __init__(self):
        self.layers = []
        self.params = {}

    def add(self, layer):
        self.layers.append(layer)

    def initialize_weights(self):
        trainable_layers = list(filter(lambda layer: issubclass(type(layer), Dense) == True, self.layers)) 
        for i in range(len(trainable_layers)):
            if hasattr(trainable_layers[i], "input_dim"):
                if i == 0:
                    self.params["w1"] = np.random.rand(trainable_layers[i].units, trainable_layers[i].input_dim)
                else:
                    raise RuntimeError("You cannot add multiple input dimensions. It is only for the first layer!")
            else:
                self.params["w"+str(i+1)] = np.random.rand(trainable_layers[i].units, trainable_layers[i-1].units)
            if trainable_layers[i].use_bias:
                self.params["b"+str(i+1)] = np.random.rand(trainable_layers[i].units, 1)
    def forward(self):
        pass

    def summary(self):
        print("-"*65)
        print("\tLayer (type)\t\tOutput Shape\t\tParam  #")
        print("="*65)
        srt = len(str(sorted(self.layers, key=lambda x:len(str(x)), reverse=True)[0]))
        for i, layer in enumerate(self.layers):
            space_factor = srt-len(str(i))-len(str(layer))
            print('\t' + " "*space_factor + str(layer) + f"-{i+1}")
            print("="*65)
        print("Total params: 0")
        print("Trainable params: 0")
        print("Non-trainable params: 0")
        print("-"*65)

model = Sequential()
model.add(Dense(32))
model.add(ReLu())
model.add(Dense(16))
model.add(Sigmoid())
model.add(Dense(8))
model.add(Tanh())
model.add(Dense(4))
model.add(LeakyReLu())

model.initialize_weights()

for i in range(4):
    print(model.params["w"+str(i+1)].shape)
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

class Input(Dense):
    def __init__(self, input_dim):
        self.layer_name = "Input"
        self.units = input_dim
        self.use_bias = False

class Gurkan(Layer):
    def __init__(self, units, gepoch = 5, type = 'one', activation = 'ReLu', use_bias = True):
        self.layer_name = "Gurkan"
        self.units = units
        self.type = type
        self.activation = activation
        self.use_bias = use_bias
        self.gepoch = gepoch

    def activate(self, x):
        if self.activation == "Linear":
            return Linear().function(x)
        if self.activation == "ReLu":
            return ReLu().function(x)
        if self.activation == "Sigmoid":
            return Sigmoid().function(x)
        if self.activation == "Tanh":
            return Tanh().function(x)
        if self.activation == "LeakyReLu":
            return LeakyReLu().function(x)

class Activation(Layer):
    pass

class Linear(Activation):
    def __init__(self):
        self.layer_name = "Linear"

    def function(self, x):
        return x

class ReLu(Activation):
    def __init__(self):
        self.layer_name = "ReLu"
    
    def function(self, x):
        return np.maximum(0, x)

class Sigmoid(Activation):
    def __init__(self):
        self.layer_name = "Sigmoid"

    def function(self, x):
        return 1/(1+np.exp(-x))

class Tanh(Activation):
    def __init__(self):
        self.layer_name = "Tanh"
    
    def function(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

class LeakyReLu(Activation):
    def __init__(self):
        self.layer_name = "LeakyReLu"

    def function(self, x):
        return np.maximum(0.01*x, x)

class Sequential:
    def __init__(self):
        self.layers = []
        self.params = {}

    def add(self, layer):
        self.layers.append(layer)

    def initialize_weights(self):
        trainable_layers = list(filter(lambda layer: issubclass(type(layer), Dense) == True, self.layers)) 
        if hasattr(trainable_layers[0], "input_dim"):
            trainable_layers.insert(0, Input(trainable_layers[0].input_dim))
            delattr(trainable_layers[1], "input_dim")
        for i in range(len(trainable_layers) - 1):
            if hasattr(trainable_layers[i+1], "input_dim"):
                raise RuntimeError("You cannot add multiple input dimensions. It is only for the first layer!")
            
            glayer = self.layers[self.layers.index(trainable_layers[i+1])-1]
            if glayer.layer_name == 'Gurkan' and glayer.type == 'one':
                self.params["w"+str(i+1)] = np.random.rand(trainable_layers[i+1].units, glayer.units)
            else:
                self.params["w"+str(i+1)] = np.random.rand(trainable_layers[i+1].units, trainable_layers[i].units)
            self.params["b"+str(i+1)] = np.random.rand(trainable_layers[i+1].units, 1)

        gurkan_layers = list(filter(lambda layer: issubclass(type(layer), Gurkan) == True, self.layers))
        for i in range(len(gurkan_layers)):
            prev_units = self.layers[self.layers.index(gurkan_layers[i])-2].units
            units = gurkan_layers[i].units
            if gurkan_layers[i].type == 'one':
                self.params["Gl"+str(i)+"w"] = np.random.rand(1, prev_units)
                for j in range(units):
                    self.params["Gl"+str(i)+"w"+str(j)] = np.random.rand(1, 1)
                    if gurkan_layers[i].use_bias == True:
                        self.params["Gl"+str(i)+"b"+str(j)] = np.random.rand(1, 1)
            elif gurkan_layers[i].type == 'multiple':
                for j in range(prev_units):
                    self.params["Gl"+str(i)+"d"+str(j)+"w"] = np.random.rand(1, 1)
                    for k in range(units):
                        self.params["Gl"+str(i)+"d"+str(j)+"w"+str(k)] = np.random.rand(1, 1)
                        if gurkan_layers[i].use_bias == True:
                            self.params["Gl"+str(i)+"d"+str(j)+"b"+str(k)] = np.random.rand(1, 1)
            else:
                raise TypeError("There are only 'multiple' and 'one' types in Gurkan layer!")

    def loss(self, A, Y):
        return np.sum(-(Y*np.log(A)+(1-Y)*np.log(1-A)))/Y.shape[1]

    def forward(self, X, Y):
        forward_params = {}
        Z = X.copy()
        A = Z.copy()
        d = 0
        g = -1
        for layer in self.layers:
            if layer.layer_name == 'Dense':
                d+=1
                Z = np.dot(self.params["w"+str(d)], A)
                if layer.use_bias == True:
                    Z = Z + self.params["b"+str(d)]
            if issubclass(type(layer), Activation) == True:
                A = layer.function(Z)
            if layer.layer_name == 'Gurkan':
                g+=1
                if layer.type == 'one':
                    forward_params["G1"] = np.dot(self.params["Gl"+str(g)+"w"], A)
                    for i in range(layer.gepoch):
                        for j in range(layer.units):
                            if j == layer.units - 1:
                                forward_params["G1"] = forward_params["G"+str(j+1)]*self.params["Gl"+str(g)+"w"+str(j)]
                                if layer.use_bias == True:
                                    forward_params["G1"]+=self.params["Gl"+str(g)+"b"+str(j)]
                                forward_params["G1"] = layer.activate(forward_params["G1"])
                            else:
                                forward_params["G"+str(j+2)] = forward_params["G"+str(j+1)]*self.params["Gl"+str(g)+"w"+str(j)]
                                if layer.use_bias == True:
                                    forward_params["G"+str(j+2)]+=self.params["Gl"+str(g)+"b"+str(j)]
                                forward_params["G"+str(j+2)] = layer.activate(forward_params["G"+str(j+2)])
                    
                    unit_layers = []
                    for x in range(layer.units):
                        unit_layers.append(forward_params["G"+str(x+1)])
                    A = np.concatenate(unit_layers)
                else:
                    prev_units = self.layers[self.layers.index(layer)-2].units
                    for unit in range(prev_units):
                        forward_params["d"+str(unit)+"G1"] = self.params["Gl"+str(g)+"d"+str(unit)+"w"]*A[unit]

                        for i in range(layer.gepoch):
                            for j in range(layer.units):
                                if j == layer.units - 1:
                                    forward_params["d"+str(unit)+"G1"] = forward_params["d"+str(unit)+"G"+str(j+1)]*self.params["Gl"+str(g)+"d"+str(unit)+"w"+str(j)]
                                    if layer.use_bias == True:
                                        forward_params["d"+str(unit)+"G1"]+=self.params["Gl"+str(g)+"d"+str(unit)+"b"+str(j)]
                                    forward_params["d"+str(unit)+"G1"] = layer.activate(forward_params["d"+str(unit)+"G1"])
                                else:
                                    forward_params["d"+str(unit)+"G"+str(j+2)] = forward_params["d"+str(unit)+"G"+str(j+1)]*self.params["Gl"+str(g)+"d"+str(unit)+"w"+str(j)]
                                    if layer.use_bias == True:
                                        forward_params["d"+str(unit)+"G"+str(j+2)]+=self.params["Gl"+str(g)+"d"+str(unit)+"b"+str(j)]
                                    forward_params["d"+str(unit)+"G"+str(j+2)] = layer.activate(forward_params["d"+str(unit)+"G"+str(j+2)])
                    unit_layers = []
                    for x in range(prev_units):
                        unit_layers.append(forward_params["d"+str(x)+"G1"])
                    A = np.concatenate(unit_layers)
        return A, self.loss(A, Y)

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
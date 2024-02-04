import numpy as np

class NeuralTensor:
    """
    A custom tensor class that supports automatic differentiation.
    """

    def __init__(self, data, requires_grad=False, creators=None, op_name=None, tensor_id=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.gradient = None
        self.tensor_id = np.random.randint(0, 100000) if tensor_id is None else tensor_id
        self.creators = creators
        self.op_name = op_name
        self.children = {}

        if creators is not None:
            for c in creators:
                if self.tensor_id not in c.children:
                    c.children[self.tensor_id] = 1
                else:
                    c.children[self.tensor_id] += 1

    def backward(self, grad=None, grad_origin=None):
        if self.requires_grad:
            if grad is None:
                grad = NeuralTensor(np.ones_like(self.data))

            if grad_origin is not None:
                if self.children[grad_origin.tensor_id] == 0:
                    raise Exception("Cannot backprop more than once")
                else:
                    self.children[grad_origin.tensor_id] -= 1

            if self.gradient is None:
                self.gradient = grad
            else:
                self.gradient += grad

            if self.creators is not None and (self._all_children_grads_accounted_for() or grad_origin is None):
                if self.op_name == "add":
                    self.creators[0].backward(self.gradient, self)
                    self.creators[1].backward(self.gradient, self)
                # Additional operations like 'sub', 'mul' etc. can be added here

    def _all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    # Define other tensor operations (__add__, __sub__, __mul__, etc.) here

class RNNLayer:
    """
    A layer in a Recurrent Neural Network.
    """

    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation = SigmoidLayer() if activation == 'sigmoid' else TanhLayer()

        self.input_hidden_layer = LinearLayer(input_size, hidden_size)
        self.hidden_hidden_layer = LinearLayer(hidden_size, hidden_size)
        self.hidden_output_layer = LinearLayer(hidden_size, output_size)

        self.parameters = (self.input_hidden_layer.get_parameters() +
                           self.hidden_hidden_layer.get_parameters() +
                           self.hidden_output_layer.get_parameters())

    def forward(self, input_tensor, hidden_tensor):
        from_prev_hidden = self.hidden_hidden_layer.forward(hidden_tensor)
        combined = self.input_hidden_layer.forward(input_tensor) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.hidden_output_layer.forward(new_hidden)
        return output, new_hidden

    def init_hidden_state(self, batch_size=1):
        return NeuralTensor(np.zeros((batch_size, self.hidden_size)), requires_grad=True)

# Define other classes like LinearLayer, EmbeddingLayer, SigmoidLayer, TanhLayer, etc.

def preprocess_shakespeare_text(file_path):
    """
    Preprocesses the Shakespeare text dataset.
    """
    with open(file_path, 'r') as file:
        raw_text = file.read()

    vocab = list(set(raw_text))
    char_to_index = {char: i for i, char in enumerate(vocab)}
    index_to_char = {i: char for i, char in enumerate(vocab)}
    indexed_data = np.array([char_to_index[char] for char in raw_text])

    return raw_text, vocab, char_to_index, index_to_char, indexed_data

def train_rnn_model(model, data, epochs=10, batch_size=32, sequence_length=100, learning_rate=0.1):
    """
    Trains the RNN model.
    """
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.get_parameters(), alpha=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        n_loss = 0

        hidden = model.init_hidden_state(batch_size)
        for batch_i in range(0, data.size(0) - sequence_length, sequence_length):
            optimizer.zero()

            hidden = NeuralTensor(hidden.data, requires_grad=True)
            loss = None

            for t in range(sequence_length):
                input = NeuralTensor(data[batch_i:batch_i + sequence_length], requires_grad=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)

                target = NeuralTensor(data[batch_i + 1:batch_i + sequence_length + 1], requires_grad=True)
                batch_loss = criterion.forward(output, target)
                if loss is None:
                    loss = batch_loss
                else:
                    loss += batch_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.data

        print(f'Epoch {epoch}, Loss: {total_loss}')

# Define other necessary functions like generate_text, etc.

# Example usage
file_path = 'tinyshakespeare.txt'
raw_text, vocab, char_to_index, index_to_char, indexed_data = preprocess_shakespeare_text(file_path)

embed = EmbeddingLayer(vocab_size=len(vocab), dim=512)
model = RNNLayer(input_size=512, hidden_size=512, output_size=len(vocab))

train_rnn_model(model, indexed_data, epochs=40, batch_size=32, sequence_length=100, learning_rate=0.05)

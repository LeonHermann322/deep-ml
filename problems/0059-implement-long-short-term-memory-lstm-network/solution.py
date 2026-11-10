import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """
        hidden = [initial_hidden_state]
        cell_state = initial_cell_state
        for step in range(x.shape[0]):
            x_step = x[step]
            gate = np.concatenate([hidden[-1], x_step[:, np.newaxis]])
            f_term = self.Wf @ gate + self.bf
            i_term = self.Wi @ gate + self.bi
            c_term = self.Wc @ gate + self.bc
            o_term = self.Wo @ gate + self.bo

            def sigmoid(inp):
                return 1 / (1 + np.exp(-inp))

            new_cell_state = sigmoid(f_term) * cell_state + sigmoid(i_term) * np.tanh(c_term)
            new_hidden_state = sigmoid(o_term) * np.tanh(new_cell_state)
            hidden.append(new_hidden_state)
            cell_state = new_cell_state

        return hidden, hidden[-1], cell_state

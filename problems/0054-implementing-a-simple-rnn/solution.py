import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    hidden = np.asarray(initial_hidden_state)
    w_x = np.asarray(Wx)
    w_h = np.asarray(Wh)
    bias = np.asarray(b)
    for step in range(len(input_sequence)):
        x = np.asarray(input_sequence[step])
        hidden_term = w_h @ hidden
        input_term = w_x @ x
        update_hidden_term = hidden_term + input_term + bias
        hidden = np.tanh(update_hidden_term)
        

    return hidden.round(4)

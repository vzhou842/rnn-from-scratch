import numpy as np

class RNN:
  def __init__(self, input_size, output_size, hidden_size=64):
    self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
    self.Wxh = np.random.randn(hidden_size, input_size) / 1000
    self.Why = np.random.randn(output_size, hidden_size) / 1000
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

  def forward(self, inputs):
    self.last_inputs = inputs
    self.last_hs = {}

    h = np.zeros((self.Whh.shape[0], 1))
    self.last_hs[-1] = h
    ys = []

    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      y = self.Why @ h + self.by

      self.last_hs[i] = h
      ys.append(y)

    return ys, h

  def backprop(self, d_out, learn_rate=1e-2):
    num_timesteps = len(self.last_inputs)

    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)

    d_Why = d_out @ self.last_hs[num_timesteps - 1].T
    d_by = d_out

    d_h = (self.Why.T @ d_out)

    for t in reversed(range(num_timesteps)):
      temp = ((1 - self.last_hs[t] ** 2) * d_h)

      d_bh += temp
      d_Whh += temp @ self.last_hs[t - 1].T
      d_Wxh += temp @ self.last_inputs[t].T

      d_h = self.Whh @ temp

    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -10, 10, out=d)

    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by

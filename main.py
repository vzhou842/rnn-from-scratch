import numpy as np
import random

from rnn import RNN
from data import *

words = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(words)
print('%d unique words found' % vocab_size)
word_to_idx = { w: i for i, w in enumerate(words) }
idx_to_word = { i: w for i, w in enumerate(words) }
print(word_to_idx)

def createInputs(text):
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

rnn = RNN(vocab_size, 2)

# Training loop
for epoch in range(1500):
  train_loss = 0
  num_correct = 0

  items = list(train_data.items())
  random.shuffle(items)
  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    outs, _ = rnn.forward(inputs)
    probs = np.exp(outs[-1]) / np.sum(np.exp(outs[-1]))

    # Calculate loss / accuracy
    train_loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    # Build dL/d(outs)
    d_L_d_outs = probs
    d_L_d_outs[target] -= 1

    rnn.backprop(d_L_d_outs)

  if epoch % 50 == 49:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %f | Accuracy: %d / %d' % (train_loss / len(train_data), num_correct, len(train_data)))

    test_loss = 0
    test_num_correct = 0
    for x, y in test_data.items():
      inputs = createInputs(x)
      target = int(y)

      outs, _ = rnn.forward(inputs)
      probs = np.exp(outs[-1]) / np.sum(np.exp(outs[-1]))

      test_loss -= np.log(probs[target])
      test_num_correct += int(np.argmax(probs) == target)

    print('Test:\tLoss %f | Accuracy: %d / %d' % (test_loss / len(test_data), test_num_correct, len(test_data)))

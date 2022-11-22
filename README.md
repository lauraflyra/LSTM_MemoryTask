# LSTM_MemoryTask

Create a LSTM that represents the chemical processes inside a neuron. This LSTM is capable of changing the neurons activation function. 

We want to observe if the neuron can learn simple memory tasks. 

Example task:

Neuron recives as input a sequence of times: [9am, 10am, 11am, 12pm, 1pm, 2pm, 3pm, 4pm] and people: [Laura, Joram, Dennis, Rob, Joram, Loreen, Felix, Mark].
After some time, the neuron receives a query: [10am] and we expect it's output to be [Joram].

The LSTM regulates the neurons activation function, which is represented by: 

gamma(x; g, theta, alpha) = g * log(1+exp(alpha*(x-theta)))/alpha

LSTM can lean g, the gain, theta, the threshold, and alpha, the softnening parameter.

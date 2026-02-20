SGNS - Word2Vec

## Implementation Details
- **Architecture:** Skip-Gram with Negative Sampling (SGNS).
- **Optimization:** Stochastic Gradient Descent (SGD) using manual gradient derivation.
- **Preprocessing:** Custom tokenization with punctuation-aware context windows to maintain semantic integrity within sentences.

## How to Run:
make sure to install all required packages stated in requirements.txt (it's only numpy)
optional: if you want to use another text than the provided, you need to put it into data/ and change the DATASET variable in .py
run the script python main.py
after the training finished, enter words in the cli you started the script in.

## Hyperparameters for Tiny Shakespeare:
Dimensions 125
Window Size 5
Negative Samples 7
Learning Rate 0.025
Epochs 15


## Task:
Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).
The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.

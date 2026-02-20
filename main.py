import numpy as np

DATASET = "data/tiny-shakespeare.txt"
DIM_COUNT = 125  # TODO find good dimension size
WINDOW_SIZE = 5  # Todo find a good window size
PUNCTUATION = [".", "!", "?", ";", ":", "--"]
LEARN_RATE = 0.025
PLOT_LOSS = False
NEGATIVE_SAMPLES = 7
EPOCHS = 15


# Reads a config file
def readConfig(dataset):
    with open(dataset, "r") as file:
        text = file.read().lower()

    for p in PUNCTUATION + ["'", ",", "--"]:
        text = text.replace(p, f" {p}")

    return text.split()

# creates a vocabulary -> each word (and punctuation) is an entry
def generateVocabulary(data):
    vocabulary = {}
    for _ in data:
        if _ not in vocabulary:
            vocabulary[_] = len(vocabulary)

    return vocabulary


def score(a, b):
    return np.dot(a, b)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def losslike(s):
    return -1 * np.log(s)

# does a single training step
def trainStep(W, W_context, idx_v, idx_u, is_positive):
    v = W[idx_v] # vector for current center word
    u = W_context[idx_u] # vector for current context word

    z = score(v, u) # dot product of u and v
    p = sigmoid(z) # sigmoid function mapping score to (0,1)

    target = 1 if is_positive else 0 # flip for error to make trainStep reusable for positivePairs and negativePairs
    error = p - target

    # gradients
    v_update = LEARN_RATE * error * u
    u_update = LEARN_RATE * error * v

    # updating vectors in matrices
    W[idx_v] -= v_update
    W_context[idx_u] -= u_update

# returns a list of words inside the window range around the center word
def getWindow(text, center_index):
    context = []

    # checks if the left side of the window is bigger than the sentence
    for offset in range(-1, -WINDOW_SIZE - 1, -1):
        j = center_index + offset
        if j < 0 or text[j] in PUNCTUATION:
            break
        context.append(text[j])

    # checks if the right side of the window is bigger than the sentence
    for offset in range(1, WINDOW_SIZE + 1):
        j = center_index + offset
        if j >= len(text) or text[j] in PUNCTUATION:
            break
        context.append(text[j])

    return context


# returns similar words compared to the word typed in by the user - this is to visualize results.
def getSimilar(word, word_map, W, top_n=5):
    if word not in word_map: return "Word not in word map" # simple check if it's even in the vocabulary


    v_center = W[word_map[word]] # find the vector of word in the matrix

    dot_prod = np.dot(W, v_center) # make the dot product with every row of W.
    # -> if vectors are pointing into the same direction the result will be high
    # -> orthogonal vectors == 0
    # -> opposite direction is negative result

    norms = np.linalg.norm(W, axis=1) * np.linalg.norm(v_center) # norm the vector to prevent wrong result because of long vecotrs
    sim = dot_prod / (norms + 1e-9) # measure the angle in between (-1,1) - use 1e-9 to prevent errors
    closest_i = sim.argsort()[-(top_n+1):-1][::-1] # filter to only display the top_n

    inv_map = {v: k for k, v in word_map.items()} # lookup the indices and return the strings
    return [inv_map[i] for i in closest_i]


def main():
    text = readConfig(DATASET) # read training file
    word_map = generateVocabulary(text) # every word 1 time
    V = len(word_map)

    W = np.random.randn(V, DIM_COUNT) * 0.01
    W_context = np.random.randn(V, DIM_COUNT) * 0.01

    for e in range(EPOCHS): # epoch = how often the training will be done
        for i, center in enumerate(text): # select current word as current center word
            if center in PUNCTUATION: # skip if it's punctuation
                continue

            context_window = getWindow(text, i) # get a list of all other words inside the window

            for w in context_window:
                trainStep(W, W_context, word_map[center], word_map[w], True) # train positive pairs consisting of elements of context and center word.

                for _ in range(NEGATIVE_SAMPLES): # Negative Sampling with NEGATIVE_SAMPLES per positive sample
                    neg_idx = np.random.randint(0, V)

                    if neg_idx == word_map[w]: continue

                    trainStep(W, W_context, word_map[center], neg_idx, False)

            # debug update
            if i % 10000 == 0:
                print(f"Epoch {e+1} | Progress: {i/len(text)*100:.1f}%")

    while True: # check training
        print("Check a word: ")
        instr = input().lower()
        if instr == "exit": break
        print(f"Similar words to '{instr}': {getSimilar(instr, word_map, W)}")


if __name__ == "__main__":
    main()

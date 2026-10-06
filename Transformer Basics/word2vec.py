import numpy as np

corpus = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat slept on the chair",
    "the dog slept on the chair",
    "the cat chased the dog",
    "the dog chased the cat",
    "the boy loved the dog",
    "the girl loved the cat",
    "the boy and girl played together",
    "the cat and dog played together",
    "the girl sat by the window",
    "the boy sat by the window",
]

EPOCHS = 1000  # You can choose the number of training epochs

tokens = []
for sentence in corpus:
    tokens.extend(sentence.split())

vocab = sorted(set(tokens))
word_to_id = {word: idx for idx,word in enumerate(vocab)}
id_to_word = {idx: word for word,idx in word_to_id.items()}

training_pairs = []
window_size = 2

for sentence in corpus:
    words = sentence.split()
    for i, center_word in enumerate(words):
        for j in range(max(0,i - window_size), min(len(words), i + window_size + 1)):
            if i == j:
                continue
            training_pairs.append((center_word, words[j]))

training_data = [(word_to_id[center_word], word_to_id[context_word]) for center_word, context_word in training_pairs]

print("Example training pairs:")
for pair in training_data[:10]:
    print(f"  center={id_to_word[pair[0]]:>8}  context={id_to_word[pair[1]]:>8}")
print()

# Each word has input embedding matrix and output embedding matrix
vocab_size = len(vocab)
embedding_dim = 5  # You can choose the dimensionality of the embeddings
learning_rate = 0.05

W_input = np.random.randn(vocab_size, embedding_dim)
W_output = np.random.randn(vocab_size, embedding_dim)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for epoch in range(EPOCHS):  # Number of training epochs
    total_loss = 0.0

    for center_id, context_id in training_data:
        center_vector = W_input[center_id]
        logits = np.dot(W_output, center_vector) # Scores using dot product with other words
        probs = softmax(logits) # Convert logits to probabilities using softmax

        target = np.zeros(vocab_size) 
        target[context_id] = 1 # True context word should have probability 1

        loss = -np.sum(target * np.log(probs + 1e-9)) # Add small value to avoid log(0)
        total_loss += loss

        # Compute gradients
        grad_logits = probs - target

        # Backpropagation to compute gradients for output and input embeddings
        # Each output word embedding gets updated based on its contribution.
        grad_W_output = np.outer(grad_logits, center_vector)

        # Gradient for the center word embedding is the sum over output words.
        # This is the classic skip-gram backprop step.
        grad_center = np.dot(W_output.T, grad_logits)

        # Update weights
        W_output -= learning_rate * grad_W_output
        W_input[center_id] -= learning_rate * grad_center

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

print("Training completed.")


# After training, the input embeddings (W_input) can be used as the word vectors.
# Similar words tend to appear close to each other in the embedding space.

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-9)  # Add small value to avoid division by zero

for word in ["cat", "dog", "boy", "girl", "mat", "window", "played", "sat"]:
    if word not in word_to_id:
        continue

    target_id = word_to_id[word]
    target_vector = W_input[target_id]

    print(f"Similar words to '{word}':")
    similarities = []
    for idx in range(vocab_size):
        if idx == target_id:
            continue
        sim = cosine_similarity(target_vector, W_input[idx])
        similarities.append((sim, id_to_word[idx]))

    similarities.sort(reverse=True)
    top_3 = similarities[:3]

    print(f"{word} is closest to : {top_3}")

print("\nWhy this matters:")
print("- Words that appear in similar contexts get similar vectors.")
print("- Example: cat and dog may cluster together because they appear around similar words.")
print("- This is the heart of Word2Vec, a widely used NLP representation learning method.")

# ------------------------------------------------------------
# 9) A simple explanation of the concept
# ------------------------------------------------------------
# In plain English:
#   If two words often appear with the same neighbors, they should have similar meanings.
#   So the model learns a vector for each word such that similar words are close together.
#   These vectors can later be used for:
#       - semantic similarity
#       - clustering
#       - text classification
#       - recommendation systems
#       - downstream NLP tasks
#
# This program is intentionally tiny and easy to read.
# Real Word2Vec implementations use larger corpora, better optimization,
# and more advanced training methods such as negative sampling or hierarchical softmax.


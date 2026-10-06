# Word2Vec Intuition: How words learn meaning from context

Word2Vec is a way to turn words into vectors, or numbers, so that similar words end up close to each other in a vector space.

The main idea is simple:

> Words that appear in similar contexts tend to have similar meanings.

For example, the words "cat" and "dog" often appear near words like "pet", "sleep", "play", and "animal". So they should get similar vector representations.

---

## 1. The problem Word2Vec solves

Traditional computers do not understand words directly. They understand numbers.

If we want a model to understand language, we need to convert each word into a vector.

A naive idea is to use one-hot encoding:

- cat = [0, 0, 1, 0, 0, ...]
- dog = [0, 1, 0, 0, 0, ...]

But this has a big problem: it treats each word as completely unrelated to every other word.

Word2Vec fixes this by learning vectors such that:

- similar words are close together
- different words are farther apart

This gives us dense, meaningful embeddings.

---

## 2. The core intuition: context reveals meaning

Look at these sentences:

- The cat slept on the couch.
- The dog slept on the couch.
- The cat chased the mouse.
- The dog chased the mouse.

The words "cat" and "dog" appear in similar positions and near similar surrounding words:

- slept
- chased
- mouse
- couch

So they probably have similar meaning.

Word2Vec uses this idea directly. It predicts a word from its context (or predicts context from a word), and while doing that it learns meaningful vectors.

---

## 3. Two main training setups: CBOW and Skip-Gram

Word2Vec has two common versions.

### A) CBOW (Continuous Bag of Words)

CBOW predicts the current word from surrounding words.

Example:

Sentence: "the cat sat on the mat"

If the center word is "sat", then the surrounding words are:

- "the", "cat", "on", "the", "mat"

The model looks at those context words and tries to predict "sat".

This is like asking:

> Given the neighboring words, what is the missing center word?

### B) Skip-Gram

Skip-Gram does the opposite.

It takes the center word and tries to predict neighboring words.

Example:

Center word: "sat"

Nearby words: "cat", "on", "the", "the", "mat"

The model is trained to output probabilities for nearby words.

Skip-gram is often more effective in practice and is the more famous formulation.

---

## 4. Why this creates meaningful vectors

The model is not trying to memorize text. It is trying to do a prediction task.

To predict a word from context, it must learn some internal representation of each word.

Those internal representations become the word embeddings.

The hidden representation gets better as the model learns which words tend to occur together.

So in the end:

- same context -> similar vectors
- similar meaning -> similar vectors
- unrelated words -> different vectors

This is the heart of Word2Vec.

---

## 5. A simple example with window size 2

Suppose we have a sentence:

"the cat sat on the mat"

Use a context window of 2.

Then for the word "sat", the context words are:

- "cat"
- "on"
- "the"
- "the"

You can form training pairs like:

- (sat, cat)
- (sat, on)
- (sat, the)

The model learns that when the word "sat" appears, nearby words usually include "cat" and "on".

Similarly, for the word "cat":

- (cat, the)
- (cat, sat)
- (cat, on)

This makes the vector for "cat" similar to the vector for words that appear in similar positions.

---

## 6. The actual math idea

A word is represented by a vector.

The model has:

- an input embedding matrix
- an output embedding matrix

For a center word, it looks up its embedding vector.

Then it computes similarity scores with all words in the vocabulary.

Those scores are transformed into probabilities by a softmax function:

$$
P(context\_word \mid center\_word) = \frac{\exp(v_{context} \cdot v_{center})}{\sum_{w \in V}\exp(v_w \cdot v_{center})}
$$

This tells the model how likely each word is to appear in the context of the center word.

The model updates its vectors to make the real context words more likely.

After many updates, the vectors become useful semantic representations.

---

## 7. Why softmax matters

Softmax converts raw dot-product scores into probabilities.

Example:

- a word with a very high score gets a high probability
- a word with low score gets low probability

This lets the model train by learning which words are likely neighbors of which center words.

---

## 8. The training objective

The model is trained using a loss function, usually cross-entropy.

The goal is:

> maximize the probability of the real context words around a center word

In other words:

- if the actual context word is "dog"
- and the model sees "cat" as center word
- then it should increase the probability of "dog" for that context

This repeats over and over for millions of examples.

---

## 9. What the learned vectors look like

After training, each word has a vector, like:

- cat = [0.22, -0.41, 0.18, ...]
- dog = [0.19, -0.38, 0.17, ...]
- lion = [0.25, -0.35, 0.12, ...]

Vector similarity can be measured with cosine similarity:

$$
\text{cosine}(a, b) = \frac{a \cdot b}{\|a\|\|b\|}
$$

This measures how aligned two vectors are.

Words with similar meaning tend to have high cosine similarity.

---

## 10. Why this is powerful

Once words are mapped to vectors, you can do many useful things:

- find similar words
- cluster words by meaning
- compare word meanings
- use embeddings as input to machine learning models
- build search and recommendation systems
- create sentence and document representations later

Example:

- king - man + woman ≈ queen

This is a famous property of word embeddings: arithmetic in vector space captures semantic relationships.

---

## 11. Important intuition to remember

The best way to think about Word2Vec is:

> The model learns word meaning by trying to predict what words tend to appear near each other.

Not by storing dictionary definitions.

Instead, it learns from distributional semantics:

- words appearing in similar contexts have similar meaning

That is the foundation of Word2Vec.

---

## 12. How to code it yourself

When coding Word2Vec from scratch, the main steps are:

1. Build a corpus of sentences
2. Tokenize the text into words
3. Create a vocabulary
4. Create center/context training pairs using a sliding window
5. Convert words to integer IDs
6. Initialize random embedding matrices
7. Compute logits for all vocabulary words
8. Apply softmax to produce probabilities
9. Compare with the true context word using cross-entropy loss
10. Backpropagate and update embeddings with gradient descent
11. Inspect the final vectors and check which words are closest

This is exactly the pattern used in the script you have in your project.

---

## 13. A very short summary

Word2Vec learns vector representations of words by solving a prediction problem over nearby words.

It does not use human-made meanings.

Instead, it discovers meaning from patterns in language itself.

That is why it is so useful: it captures semantic similarity in a numerical form.

---

## 14. Final takeaway

If you remember only one sentence, remember this:

> Word2Vec turns words into vectors by learning that words sharing similar contexts have similar meanings.

That is the complete intuition.

---

If you want, I can next turn this into a more step-by-step coding guide with:

- a minimal hand-written example
- pseudocode for the training loop
- a clearer explanation of softmax and gradient descent
- or a simpler implementation using only Python lists and loops

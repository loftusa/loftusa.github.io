## Extending Bradley Terry


### The Bradley-Terry Model

The Bradley-Terry model assigns a strength parameter $\theta_i$ to each item $i$. The probability that item $i$ wins against item $j$ is:

$$P(i > j) = \sigma(\theta_i - \theta_j) = \frac{1}{1 + e^{-(\theta_i - \theta_j)}}$$

where $\sigma$ is the logistic sigmoid function. The outcome depends on the *difference* in strengths between competitors.

### The Loss Function

We fit the model by minimizing the binary cross-entropy loss (negative mean log-likelihood). For $N$ matchups with labels $y_i$ and predicted probabilities $p_i$:

$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

This loss penalizes incorrect predictions: the term $y_i \log(p_i)$ contributes when model A wins ($y_i=1.0$), while $(1 - y_i) \log(1 - p_i)$ contributes when model A loses ($y_i=0.0$). For ties ($y_i=0.5$), both terms contribute equally.



## Dataset

The dataset contains human preference judgments from pairwise comparisons of AI model responses. Each comparison involves:
- A user submitting a prompt
- Two different AI models generating responses
- The user selecting which response they prefer (or marking a tie)

### Data Format

A `load_data()` function is provided which loads and splits the data into train and eval datasts both with the following key fields:

* `models`: List of strings (length M) containing all unique model names, ordered alphabetically
* `matchups`: [N, 2] torch tensor of integers representing pairs of models being compared. Each integer is an index into the `models` list
* `outcomes`: [N] torch tensor of floats with values in {1.0, 0.0, 0.5} indicating whether the first model won (1.0), lost (0.0), or tied (0.5)
* `users`: List of strings containing all unique user identifiers
* `user_ids`: [N] torch tensor of integers, where each integer represents a unique user who made the judgment
* `prompt_embs`: [N, 32] torch tensor of float embeddings representing the semantic content of each prompt
* `response_a_embs`: [N, 32] torch tensor of float embeddings for the first model's response
* `response_b_embs`: [N, 32] torch tensor of float embeddings for the second model's response

All embeddings are 32-dimensional vectors obtained from a toy embedding model.

Examples:
If there are the following battles:

* Model A vs Model B, judged by User X who voted for Model A
* Model B vs Model C, judged by User Y who voted for Model C
* Model C vs Model D, judged by user X who voted tie

The data would look like:
```
train_data, val_data, models, users = load_data()

models
# ["Model A", "Model B", "Model C", "Model D"]

users
# ["User X", "User Y"]

train_data["matchups"]
# [[0, 1],    # A vs B
   [1, 2],    # B vs C
   [2, 3]]    # C vs D

train_data["outcomes"]
# [1.0,    # model in first column won
   0.0,    # model in second column won
   0.5]    # tie

train_data["user_ids"]
[0,    # "User X" is index 0 in users
 1,    # "User Y" is index 1 in users
 0]    # "User X" is index 0 in users
```

# Tasks

## 1. Bradley-Terry Warmup
Use the given started code to implement the loss funtion and Bradley-Terry model.

Run it on the sample data

## 2. User-Model Interaction Extension
Propose and implement a way to use the user_id information to create a more expressive extension of BT.


## 3. Content Represenation Extension
Propose and implement a method which utilizes the provided embeddings to model the vote outcome.


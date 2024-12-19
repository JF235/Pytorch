# CH3 - Classification

No último capítulo, fizemos um exemplo de regressão. Agora, classificação...

## Classification

F(Input) = Class

- Binary, Class in {0, 1}
- Multi-class, Class in {0, 1, 2, ..., n}
- Multi-label, Class in $\mathbb{Z}^m$

## Linear Layer

$$y = xA^T + b$$

Se `in_features` = 2 e `out_features` = 8, então:
- $\text{shape}(x) = (1, 2)$
- $\text{shape}(A) = (8, 2)$, $\text{shape}(A^T) = (2, 8)$
- $\text{shape}(b) = (1, 8)$

$$(1, 2)\odot(2, 8) + (1, 8) = (1,8)$$

## Logit

Raw output from the model, before applying the activation function.

Logits $\to$ Probability Distribution $\to$ Class

## Non-linear Activation Functions

- Types (ReLU, Sigmoid, Tanh)
- Why non-linear? XOR problem (auth: Minsky & Papert, 1969)

## Cross Entropy

## Classification Metric

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
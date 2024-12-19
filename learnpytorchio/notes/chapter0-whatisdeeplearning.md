# CH0 - What Is Deep Learning?

Turning data into numbers -> Finding pattern in numbers

## Artificial Intelligence vs. Machine Learning vs. Deep Learning

AI > ML > DL

1. AI - Encontrar regras/algoritmos que repliquem um comportamento inteligente e obtenham os resultados desejados.
2. ML - Dado um conjunto de entradas e saídas, usar algoritmos que encontrem regras que mapeiam as entradas nas saídas de forma automática.

Por que usar ML? Quando todas as regras não são conhecidas ou são muito complexas para serem escritas manualmente.

Existem limitações: Se você pode escrever um sistema baseado em regras que resolva o problema e não precisa de ML, faça isso.

## Escopo e Limitações

Escopo:
- Problemas com listas extensas de regras
- Ambientes que mudam com o tempo (precisam de adaptação)
- Quando existem um grande conjunto de dados

Não recomendados:
- Fortemente baseado em explicabilidade
- Quando a forma tradicional resolve
- Quando erros são inaceitáveis
- Quando há poucos dados

## Focando em: ML vs. DL

ML dados estruturados (Gradient Boosted Machine, XGBoost) vs. DL dados não estruturados (Redes Neurais)

Dados estruturados ("shallow algorithms")
- Random forest
- Gradient boosted models
- Naive Bayes
- Nearest neighbors
- Support vector machines (SVM)

Dados não estruturados ("deep learning")
- Neural networks
- Fully connected networks
- Convolutional neural networks (CNN)
- Recurrent neural networks (RNN)
- Transformers
  
## Neural Networks

Inputs -> Numerical Encoding -> Learns Representation (pattern/features/weights) -> Representation output -> Output

Anatomia da rede neural:
- Neurônios
- Conexões
- Camadas (funções lineares + não lineares)

Arquitetura:
- Número e tipo de camadas

## Types of Learning

1. Supervised Learning: Dados rotulados (entradas e saídas esperadas)
2. Unsupservised Learning: Dados não rotulados (aprendizado de padrões)
3. Self-supervised Learning: Dados não rotulados, mas com alguma informação (ex: prever a próxima palavra em uma frase)
4. Transfer Learning: Usar um modelo pré-treinado para um problema similar
5. Reinforcement Learning: Aprender a partir de recompensas

## Aplicações

- Recomendação
- Tradução e Reconhecimento de Fala (seq2seq: sequence-to-sequence)
- Computer Vision and Natural Language Processing (regression/classification)
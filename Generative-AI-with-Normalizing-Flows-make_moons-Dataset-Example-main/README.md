# Generative-AI-with-Normalizing-Flows-make_moons-Dataset-Example
In this repository, I have worked to demonstrate an example of applying the concept of normalizing flows in generative AI to the make_moons dataset. Normalizing flows are a powerful technique for generative modeling, and this project demonstrates their application to a noisy 2D dataset

# The Mysterious Shop of JACOB'S

## Introduction
As you wander through a quaint village, your attention is drawn to a peculiar shop with a sign that reads "JACOB'S." Curiosity gets the better of you, and you cautiously step inside. There, you encounter an elderly man behind the counter. Intrigued, you inquire about the nature of his wares.

The old man, with a mysterious glint in his eye, reveals that he offers a service for digitizing paintings, but there's a unique twist. After a moment of rummaging in the back of the shop, he produces a silver box adorned with the letters F.L.O.W. He explains that these letters stand for "Finding Likenesses Of Watercolors," giving you a rough idea of what the machine does. Intrigued, you decide to give it a try.

## The Unusual Machine
Returning the next day, you hand over a collection of your cherished paintings to the shopkeeper. He feeds them into the F.L.O.W. machine, which springs to life with a symphony of hums and whistles. After a while, the machine produces a list of seemingly random numbers. The shopkeeper hands you the list and begins to tot up the charges for the digitization service and the F.L.O.W. box.

Perplexed by the list of numbers and how to retrieve your beloved paintings, you inquire about the next steps. The shopkeeper, wearing an exasperated expression, walks back to the machine. This time, he feeds the list of numbers into the machine from the opposite side. The machine whirs once more, and you watch in bewilderment until, finally, your original paintings emerge from where they were first introduced.

## A Unique Twist
With your precious paintings safely back in your possession, you contemplate storing them in the attic. Before you can depart, the shopkeeper guides you to another corner of the shop, where a massive bell hangs from the rafters. He takes a colossal stick and strikes the bell, sending vibrations reverberating throughout the store.

Instantaneously, the F.L.O.W. machine you carry under your arm starts to hiss and whirl in reverse, as if new numbers have been inserted. After a brief pause, an array of stunning watercolor paintings begins to cascade out of the machine. These paintings retain the style and form of your original set but are entirely unique.

You question the shopkeeper about the workings of this miraculous device. He elucidates that the secret lies in a special process he has developed, ensuring that the transformation is lightning-fast and simple to calculate. Simultaneously, it's sophisticated enough to convert the vibrations caused by the bell into the intricate patterns and shapes present in the paintings.

## The Exit
Recognizing the incredible potential of this contraption, you promptly settle the bill for the device and make your exit from the shop. You're elated that you now have a means to generate fresh paintings in your favored style. All it takes is a visit to the shop, a melodious chime of the bell, and a bit of patience while your F.L.O.W. machine works its magic!

This is the tale of the mysterious shop of JACOB'S, where artistry and innovation come together in a delightful and unique way.

---

# Understanding Probability Distribution Transformations with the Jacobian Determinant

## The Starting Point
- We begin with a probability distribution, pX, defined over a two-dimensional rectangular region represented by coordinates x1 and x2.
- To be a valid probability distribution, it must integrate to 1 over its entire domain, ensuring it represents a meaningful probability function.

## The Transformation Goal
- Our objective is to change this probability distribution to be defined over a different region, such as a unit square Z.
- This transformation is achieved through a function f, which maps points from X to Z. The transformation equations are as follows:
   - z1 = (x1 - 1) / 3
   - z2 = x2 / 2

## The Inverse Transformation
- It is crucial that the transformation from X to Z is invertible. In other words, there must be a function g that can map points back from Z to X. This is necessary to maintain consistency between the two spaces.

## The Challenge
- We face a challenge when we try to understand how the probability distribution pX changes as we move from X to Z.
- The issue becomes apparent when we integrate pX in the new space Z. It no longer equals 1, and this problem is related to the size difference between X and Z.

## Integration and Probability Distribution
- The integral of a probability distribution over its domain represents the total probability in that space.
- In our starting point, ∫∫<sub>X</sub> pX(x) dx1 dx2 = 1, which ensures that the total probability in X equals 1.

## The Jacobian Matrix
- The Jacobian of the function z = f(x) is a matrix that contains the first-order partial derivatives of the transformation equations. In our context, the Jacobian matrix is structured as follows:
   | ∂z1/∂x1   ∂z1/∂x2 |
   | ∂z2/∂x1   ∂z2/∂x2 |

## The Role of the Jacobian Determinant
- The Jacobian determinant, often denoted as |J(f)|, is a mathematical tool that quantifies how the area or volume changes during the transformation.
- It tells us how much the probability mass is either "stretched" or "compressed" as we move from X to Z.

## Calculating the Determinant
- The determinant of the Jacobian matrix is calculated using the following formula:
   det |∂z1/∂x1   ∂z1/∂x2 |
       |∂z2/∂x1   ∂z2/∂x2 |
- In our specific example, the partial derivatives for our transformation are as follows:
   - ∂z1/∂x1 = 1/3
   - ∂z1/∂x2 = 0
   - ∂z2/∂x1 = 0
   - ∂z2/∂x2 = 1/2

## Determinant Calculation
- Applying these partial derivatives to the determinant formula:

  det |1/3  0 |
      |0    1/2|
   = (1/3 * 1/2) - (0 * 0) = 1/6

## Interpreting the Determinant
- In the given context, the determinant (1/6) is a scaling factor that compensates for the size difference between X and Z. It ensures the probability distribution remains valid and consistent in the transformed space.

## Absolute Value of the Determinant
- The determinant can be positive or negative. Taking the absolute value ensures we consider the relative change in volume or area without regard to the direction of the transformation.
![Expaination of Probablity Distribution](https://github.com/prateekghorawat/Generative-AI/blob/main/Generative-AI-with-Normalizing-Flows-make_moons-Dataset-Example-main/Images/Jabobian_Graph.png)


---

# Understanding Probability Distribution Transformations
![realnvp.png](https://github.com/prateekghorawat/Generative-AI/blob/main/Generative-AI-with-Normalizing-Flows-make_moons-Dataset-Example-main/Images/formula.png)
## Equation: The Change of Variables Equation
- **Equation** is a fundamental tool in probability theory, describing how a probability distribution, pX, over variables x, relates to a transformed distribution, pZ, over variables z.

- It incorporates the determinant of the Jacobian matrix, denoted as **det(∂z/∂x)**, which quantifies how space is transformed.

## Generative Model Building
- The core idea is to find a simple and easy-to-sample distribution, **pZ**, such as a Gaussian distribution.

- To achieve data generation in the original domain, X, we need two functions:
  1. **Forward Mapping (f):** This function, **f(x)**, maps data points from the original space X to the transformed space Z.
  2. **Inverse Mapping (g):** This function, **g(z)**, maps sampled points from Z back to X.

- Using these functions, we can sample z from pZ and use the inverse function g(z) to generate data x in the original domain X.

- **Equation 6-1** allows us to compute the data distribution pX(x) using the probability distribution pZ(z) and the determinant of the Jacobian matrix.

## Practical Challenges
1. **High-Dimensional Determinants:**
   - Calculating the determinant of a high-dimensional Jacobian matrix (∂z/∂x) is computationally expensive.
   - For complex datasets with many dimensions, the computational cost becomes prohibitive.
   - For instance, even small 32x32 pixel grayscale images can have 1,024 dimensions, making determinant calculations infeasible.

2. **Invertible Functions:**
   - Finding an invertible function f(x), meaning it can be easily reversed to find x from z, is challenging.
   - Traditional neural networks, widely used in deep learning, are not naturally designed to be inverted. They typically work in one direction, from input to output.

## RealNVP Transformations
- To overcome these challenges, specialized neural network architectures like **RealNVP (real-valued non-volume preserving) transformations** are employed.

- RealNVP ensures that the change of variables function f(x) is invertible, allowing the derivation of the corresponding inverse function g(z) for the reverse transformation.

- Additionally, RealNVP architectures are designed in a way that makes it computationally feasible to calculate the determinant of the Jacobian matrix, even for high-dimensional data.

---

# Understanding Regularization with L2 Regularization

## What is Regularization?
- Regularization is a technique used in machine learning to prevent overfitting.
- Overfitting occurs when a model fits the training data too closely, including noise in the data, which leads to poor generalization on new, unseen data.

## L2 Regularization (Weight Decay):
- L2 regularization is a common type of regularization that discourages large weights in a model.
- It's also known as "weight decay."
- Large weights can make models overly complex and prone to overfitting.
- L2 regularization adds a penalty term to the loss function during training to control the size of weights.

---

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

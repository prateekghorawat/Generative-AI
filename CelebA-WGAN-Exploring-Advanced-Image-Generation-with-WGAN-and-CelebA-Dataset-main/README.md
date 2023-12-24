# CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset
A project implementing a Wasserstein Generative Adversarial Network (WGAN) for training and generating high-quality images using the CelebA dataset. Explore the power of GANs in generating celebrity faces and use the Wasserstein distance for more stable training and improved image quality.

---

DataSet used is available on Kaggle as  celeba

---

### **Reason for Scaling data and that too to a particular SCALE..**
The original data is scaled in the range [0, 255] to denote the pixel intensity. When
training GANs we rescale the data to the range [–1, 1] so that we can use the tanh
activation function on the final layer of the generator, which tends to provide stronger gradients than the sigmoid function

---


### **Goal of Discriminator**
The goal of the discriminator is to predict if an image is real or fake. This is a supervised image classification problem

---

We used a stride of 2 in some of the Conv2D layers to reduce the spatial
shape of the tensor as it passes through the network (64 in the original image, then
32, 16, 8, 4, and finally 1), while increasing the number of channels (1 in the grayscale
input image, then 64, 128, 256, and finally 512), before collapsing to a single
prediction.
We use a sigmoid activation on the final Conv2D layer to output a number between 0
and 1.

---

### **Explain the Parameters of LeakyReLU.**
Imagine you have a magic box that takes any number as an input and transforms it. For most positive numbers, the box does exactly what you'd expect. If you put in 5, you get 5 back. If you put in 10, you get 10 back.

However, there's something special about this box when it comes to negative numbers. Instead of turning them into 0 (like a normal box would), this box only slightly changes them. If you put in -5, you don't get 0, you get -4.5 back. If you put in -10, you don't get 0; you get -9 back.

The "0.9" is like a rule for this box, saying that it will keep 90% of the negative value. So, if you put in -5, it retains 90% of it, which is -4.5. This helps the box avoid completely erasing negative information.

Here's a simple formula for this magic box:

If you put in a positive number (e.g., 5), you get the same positive number (5) back.
If you put in a negative number (e.g., -5), you get 90% of that negative number back (90% of -5 is -4.5).
This "magic box" with the "0.9" rule is similar to how the Leaky ReLU activation function works, allowing some information to pass through for negative inputs, rather than completely erasing it. It's used in neural networks to help the network learn better, especially when dealing with certain types of data or problems.


---

### **Explain the Parameters of BatchNormalization.**
Imagine a Group of Chefs Baking Cookies:

You have a group of chefs, each baking cookies. In the beginning, they're all working at different speeds and using different-sized bowls. Some chefs are faster, and some are slower. This leads to inconsistent results.

Batch Normalization (BN) as a Cookie Factory Manager:

Now, you decide to introduce a "cookie factory manager" named Batch Normalization. The role of this manager is to make sure all the chefs work at a similar pace and use bowls of the same size. The manager has two important tasks:

Scaling the Cookie Dough: The manager scales the amount of cookie dough each chef is working with. If one chef was using double the amount of dough compared to another chef, the manager helps them all use roughly the same amount. This ensures consistent dough usage.

Shifting the Start Time: The manager also shifts the start time of each chef's baking process. If one chef started baking much earlier or later than the others, the manager helps them all start around the same time. This ensures consistent starting conditions.

Parameter: Momentum

The "momentum" in Batch Normalization is like a manager's decision-making speed. A high momentum means the manager makes changes more slowly, while a low momentum means the manager adapts quickly to the chefs' baking speeds.

High Momentum (e.g., 0.9): The manager makes gradual changes over time, allowing the chefs to maintain some of their original styles.

Low Momentum (e.g., 0.1): The manager quickly enforces strict consistency, and the chefs start baking in a very similar way.

Example: Low Momentum (0.1)

If the manager has low momentum (0.1), it's like having a strict manager who ensures all chefs work almost exactly the same way. They all use almost the same amount of dough and start baking around the same time. This level of consistency can lead to very similar, but high-quality cookies.

Example: High Momentum (0.9)

If the manager has high momentum (0.9), it's like having a more lenient manager who allows some variation. Chefs may still have a bit of their unique styles, but they are somewhat adjusted to work more similarly. This can be useful when you want to preserve some individual characteristics in the final cookies.

In Batch Normalization, the "momentum" parameter determines how quickly the manager adjusts the scaling and shifting to create consistency among the data. The choice of momentum depends on the problem and data characteristics, and it can impact the trade-off between fast convergence and preserving some originality in the data.

---

#### **Theory for GAN'S.**
* **The key to understanding GANs lies in understanding the training process for
the generator and discriminator.**
* **We can train the discriminator by creating a training set where some of the images are real observations from the training set and some are fake outputs from the generator**
* **We then treat this as a supervised learning problem, where the labels are 1 for the real images and 0 for the fake images, with binary cross-entropy as the loss function.**

* **How should we train the generator?**
  * we have a discriminator that does exactly that! We can generate a batch of images and pass these
  through the discriminator to get a score for each image. The loss function for the
  generator is then simply the binary cross-entropy between these probabilities and vector of ones, because we want to train the generator to produce images that the discriminator thinks are real.
  * We must alternate the training of these two networks, making sure that we
  only update the weights of one network at a time.
* **For example, during the generator
training process, only the generator’s weights are updated. If we allowed the discriminator’s weights to change as well, the discriminator would just adjust so that it is more
likely to predict the generated images to be real, which is not the desired outcome.
We want generated images to be predicted close to 1 (real) because the generator is
strong, not because the discriminator is weak.**


---

![image](https://github.com/prateekghorawat/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset/blob/main/Images/photo1.png)

---

1. The loss function for the generator and discriminator is BinaryCrossentropy.
2. To train the network, first sample a batch of vectors from a multivariate standard
normal distribution.
3. Next, pass these through the generator to produce a batch of generated images.
4. Now ask the discriminator to predict the realness of the batch of real images…
…and the batch of generated images.
5. The discriminator loss is the average binary cross-entropy across both the real
images (with label 1) and the fake images (with label 0).
6. The generator loss is the binary cross-entropy between the discriminator predictions for the generated images and a label of 1.

---

### **What is Beta in Adam?**
Imagine you're trying to find the best path to reach a treasure buried underground. You have a map that shows how steep or smooth the terrain is.

* **beta_1** is like your memory of recent steps. If you have a good memory (beta_1 close to 1), you'll remember your last few steps very well. If the terrain suddenly becomes steep, you'll quickly adjust your direction to find the treasure.

* **beta_2** is like your awareness of the ground's roughness. If you're very aware (beta_2 close to 1), you'll notice even slight bumps in the terrain. This helps you avoid digging in very rocky or uneven areas.

Now, let's say you're walking with these two parameters:

You have a moderate memory (beta_1 = 0.5): You somewhat remember your last few steps but not too well.

You're highly aware of the ground (beta_2 = 0.999): You notice even tiny bumps in the terrain.

---

Steps in a Generative Adversarial Network (GAN):

**Training Steps in a GAN:**

1. **Generate Random Latent Vectors:**
   - GANs start by generating random latent vectors. These vectors are typically drawn from a standard normal distribution and serve as input to the generator.

2. **Generator Generates Fake Images:**
   - The generator model uses these latent vectors to generate fake (synthetic) images.
   - The generator's goal is to create fake images that are convincing and resemble real data.

3. **Discriminator Evaluates Real Images:**
   - The discriminator model evaluates real images (from the dataset) and predicts whether they are real or fake.
   - The discriminator's goal is to get better at distinguishing real from fake images.

4. **Discriminator Evaluates Fake Images:**
   - The discriminator also evaluates the fake images generated by the generator.
   - It aims to classify these fake images as "fake" accurately.

5. **Prepare Labels for Real and Fake Images:**
   - For real images, labels are set to 1 (indicating "real").
   - For fake images, labels are set to 0 (indicating "fake").

6. **Add Noise to Labels (Optional):**
   - To prevent the discriminator from becoming overly confident too quickly, you can add small random noise to the labels. This introduces some uncertainty.
   - Example: Real label 1 might become 1.05 with noise.

7. **Calculate Discriminator Loss (Real):**
   - Using the labels (possibly with added noise) and the discriminator's predictions for real images, calculate the discriminator loss.
   - Commonly used loss function: Binary Cross-Entropy Loss.

8. **Calculate Discriminator Loss (Fake):**
   - Similarly, calculate the discriminator loss for the fake images.
   - The loss measures how well the discriminator can classify fake images.

9. **Discriminator's Objective:**
   - The discriminator aims to minimize these loss values.
   - The combined discriminator loss represents how well it can distinguish real from fake images.

10. **Calculate Generator Loss:**
    - The generator loss measures how well the generator is at producing fake images that can "fool" the discriminator.
    - It compares the fake image predictions with the ideal "real" labels (set to 1).

11. **Update Discriminator and Generator:**
    - Gradients of the discriminator and generator losses are computed.
    - The gradients are used to update the parameters (weights) of both the discriminator and generator models.

12. **Repeat for Each Batch:**
    - These steps are repeated for each batch of data during training.
    - Over time, both the discriminator and generator improve their performance.

13. **Repeat Training Epochs:**
    - Training typically involves multiple epochs, where the entire dataset is processed multiple times.
    - The goal is to improve the generator's ability to create convincing images.

14. **Noise Reduction (Over Time):**
    - As training progresses, the noise in labels may be reduced to encourage more accurate predictions.

15. **Convergence:**
    - Over many training iterations, the generator improves its ability to generate more realistic images.
    - The discriminator becomes better at distinguishing real from fake.

16. **Balance and Improve:**
    - The GAN training process continues until a balance is reached, where the generator produces high-quality images, and the discriminator is a strong evaluator.

This process continues iteratively until the generator can produce convincing, high-quality images, making the GAN a powerful tool for generating data.

---

### **Why we need the compare function and how does it work?**

The L1 distance, also known as the Manhattan distance, is a way to calculate the difference between two images pixel by pixel. It works by taking the absolute difference between the pixel values of the two images and then averaging these absolute differences. This gives you a single number that represents how different the two images are.

Here's a simple example to illustrate this:

Imagine we have two grayscale images, A and B, both with the same dimensions. Each image is a grid of pixels, and each pixel has a numerical value that represents its brightness. To calculate the L1 distance between these images, you would do the following:

For each corresponding pixel in A and B, calculate the absolute difference between their values (i.e., the absolute value of A - B).

Add up all these absolute differences for all the pixels.

Divide this sum by the total number of pixels to get the average absolute difference.

The result is a single number that tells you how different the two images are. A lower L1 distance means the images are more similar, and a higher L1 distance means they are more dissimilar.

So, when you use the compare_images function with two images as input, it calculates the L1 distance between them and returns the average absolute difference. You can use this distance measure to assess how closely a generated image resembles the nearest image in the training set. If the L1 distance is low, it means the generated image is similar to the training data, and if it's high, it indicates a significant difference.

This measure helps ensure that your generative model is not simply copying training images but is capable of producing novel and diverse outputs.


---

### **Issues and there Solutions**

**Discriminator overpowers the generator :**

If the discriminator becomes too strong, the signal from the loss function becomes
too weak to drive any meaningful improvements in the generator.

If discriminator loss function collapsing, we need to find ways to
weaken the discriminator.

Try the following suggestions:
* Increase the rate parameter of the Dropout layers in the discriminator to
dampen the amount of information that flows through the network.
* Reduce the learning rate of the discriminator.
* Reduce the number of convolutional filters in the discriminator.
* Add noise to the labels when training the discriminator.
* Flip the labels of some images at random when training the discriminator.

---

**Generator overpowers the discriminator :**

If the discriminator is not powerful enough, the generator will find ways to easily trick the discriminator with a small sample of nearly identical images. This is known as mode collapse.

For example, suppose we were to train the generator over several batches without updating the discriminator in between. The generator would be inclined to find a single observation (also known as a mode) that always fools the discriminator and would
start to map every point in the latent input space to this image. Moreover, the gradients of the loss function would collapse to near 0, so it wouldn’t be able to recover from this state.
Even if we then tried to retrain the discriminator to stop it being fooled by this one point, the generator would simply find another mode that fools the discriminator,
since it has already become numb to its input and therefore has no incentive to diversify its output

If you find that your generator is suffering from mode collapse, you can try strengthening the discriminator using the opposite suggestions to those listed in the previous
section. Also, you can try reducing the learning rate of both networks and increasing the batch size.

---

**Uninformative loss :**
Since the deep learning model is compiled to minimize the loss function, it would be
natural to think that the smaller the loss function of the generator, the better the quality of the images produced. However, since the generator is only graded against the
current discriminator and the discriminator is constantly improving, we cannot compare the loss function evaluated at different points in the training process. Indeed, in
Figure 4-6, the loss function of the generator actually increases over time, even
though the quality of the images is clearly improving. This lack of correlation
between the generator loss and image quality sometimes makes GAN training difficult to monitor.

---

**Hyperparameters :**
As we have seen, even with simple GANs, there are a large number of hyperparameters to tune. As well as the overall architecture of both the discriminator and the generator, there are the parameters that govern batch normalization, dropout, learning
rate, activation layers, convolutional filters, kernel size, striding, batch size, and latent
space size to consider. GANs are highly sensitive to very slight changes in all of these
parameters, and finding a set of parameters that works is often a case of educated trial
and error, rather than following an established set of guidelines.

---

In recent years, several key advancements have drastically improved the overall stability of GAN models and diminished the likelihood of some of the problems listed
earlier, such as mode collapse.

---

### **Gradient Penalty**
* It is used to add the contrain to WGAN
* The gradient penalty loss is a technique used in training machine learning models, particularly in Wasserstein GANs (WGANs), to encourage the model to generate more realistic images. It does this by making sure that the model doesn't make extreme changes in its predictions between nearby images.
---
* Here's a simple explanation with an example:

    Imagine you're training a GAN to generate realistic images of cats. The generator in the GAN creates images, and the discriminator tries to determine whether these images are real (from a dataset of real cat images) or fake (generated by the GAN).


Now, let's talk about the gradient penalty:

* Gradient: The gradient measures how much the model's predictions change as you make small changes to the input images. In this case, it's like asking, "If we make a tiny adjustment to a cat image, how much will the model's guess change?"

* Lipschitz Constraint: The Lipschitz constraint is like a rule that says the model should not make extremely wild changes in its predictions. This helps ensure stability in the training process.

* The WGAN-GP technique aims to encourage the model to follow this constraint. Here's how it works:

  * Instead of calculating the gradient for every possible image, which would be very slow, it only evaluates the gradient at a few carefully chosen points.
  * These points are created by taking pairs of real and fake images (real cat images and fake cat images created by the generator) and interpolating between them. Interpolation means creating new images that are a mix between a real image and a fake image.
  For example, let's say you have one real cat image and one fake cat image:

* You can create interpolated images that are a blend of these two, like a cat image that's 20% real and 80% fake, another that's 50% real and 50% fake, and so on.
The WGAN-GP method calculates the gradient at these interpolated points. If the gradient is too extreme (meaning a small change in the input results in a big change in the model's prediction), it penalizes the model.

* By doing this, the model is encouraged to have a smoother transition between real and fake images, which helps in generating more realistic and coherent images without dramatic changes from one image to the next. This makes the training process more stable and efficient.

![image](https://github.com/prateekghorawat/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset/blob/main/Images/photo2.png)
![image](https://github.com/prateekghorawat/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset/blob/main/Images/photo3.png)

---

## Calculating Gradient Penalty in GAN

In a Generative Adversarial Network (GAN), the gradient penalty is a crucial technique to encourage a smoother transition between real and fake images. This code snippet explains how to calculate the gradient penalty and why it's important.

### Background
- In a GAN, the generator creates fake images, and the discriminator (or critic) distinguishes between real and fake images.
- The gradient penalty is used to prevent the critic from making extreme predictions and to ensure a smooth transition between real and fake images. This helps stabilize training and improve the quality of generated images.

### Code Explanation

1. `alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)`:
   - This line generates random values (alphas) for each image in a batch.
   - Alphas range from 0 to 1, e.g., 0.2, 0.7, etc. These values determine how much of each image is real or fake when creating interpolated images.

2. `diff = fake_images - real_images`:
   - Calculates the difference between the fake and real images pixel by pixel.
   - Measures how much the generated fake images differ from the real ones.

3. `interpolated = real_images + alpha * diff`:
   - Creates new images (interpolated) by blending real and fake images using alphas.
   - Example: If you have a real face image and a fake face image with an alpha of 0.2, the interpolated image is 20% real and 80% fake.

4. `gp_tape.watch(interpolated)`:
   - Informs TensorFlow to pay attention to interpolated images to calculate gradients (rate of change) with respect to these images.

5. `pred = self.critic(interpolated, training=True)`:
   - Uses the critic model to predict the authenticity of interpolated images.
   - The critic's job is to determine how real or fake an image is.

6. `grads = gp_tape.gradient(pred, [interpolated])[0]`:
   - Calculates how the critic's predictions change as you make small changes to the interpolated images.
   - Computes the gradient.

7. `norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])`:
   - Measures the magnitude (norm) of gradients across the dimensions of the image.
   - This indicates how big these gradients are.

8. `gp = tf.reduce_mean((norm - 1.0) ** 2)`:
   - Calculates the gradient penalty.
   - Penalizes the model if the gradient norm deviates from the ideal value of 1, enforcing a smoother transition between real and fake images.

### Conclusion
In the context of GAN training, the gradient penalty ensures that the critic doesn't make extreme predictions and encourages the production of more realistic and coherent images by the generator. It's a crucial technique for stable and high-quality GAN training.

---

### **Batch Normalization in a WGAN-GP**
One last consideration we should note before training a WGAN-GP is that batch normalization shouldn’t be used in the critic. This
is because batch normalization creates correlation between images
in the same batch, which makes the gradient penalty loss less effective. Experiments have shown that WGAN-GPs can still produce
excellent results even without batch normalization in the critic.

---

### **Key differences between a standard GAN and a WGAN-GP**
• A WGAN-GP uses the Wasserstein loss.

• The WGAN-GP is trained using labels of 1 for real and –1 for fake.

• There is no sigmoid activation in the final layer of the critic.

• Include a gradient penalty term in the loss function for the critic.

• Train the critic multiple times for each update of the generator.

• There are no batch normalization layers in the critic.

---

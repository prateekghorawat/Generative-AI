# Hugging Face: My Journey to Generative AI using Open Source Models

This repository is the result of my hands-on exploration and learning journey with Hugging Face—one of the world’s leading platforms for natural language processing, generative AI, and modern machine learning. Through this notebook, I demonstrate a range of tasks, experiments, and best practices that showcase my understanding of both foundational and advanced AI concepts.

## What’s Inside

This project is a comprehensive, example-driven guide covering the following topics:

- **Pipeline Fundamentals:**
Applied Hugging Face’s flexible pipelines for rapid prototyping and experimentation, including sentiment analysis, text generation, translation, and zero-shot classification.
- **Task Coverage:**
    - Sentiment Analysis
    - Named Entity Recognition
    - Question Answering
    - Text Summarization
    - Text Classification \& Zero-shot Classification
    - Text Generation
    - Machine Translation
    - Image Generation (Stable Diffusion)
- **Tokenizers:**
Explored different tokenizer architectures and compared how top models encode and decode text.
- **Model Comparison:**
Loaded and evaluated multiple instruct-tuned and base models side-by-side—demonstrating transferability and differences in tokenization and inference.
- **Quantization:**
Implemented quantization using `BitsAndBytesConfig` to optimize models for lower memory usage, higher speed, and deployment on resource-constrained devices.
- **Chat/Instruction-Tuned Models:**
Leveraged chat-template utilities for structuring prompts and interfacing with advanced "Instruct" models in conversational scenarios.
- **Best Practices:**
Automated device placement, managed memory, and wrapped workflows as reusable functions for modularity.


## My Learning Goals \& Achievements

- **From Beginner to Practitioner:**
I started with basic tasks and advanced all the way to model quantization and prompt engineering.
- **End-to-End Understanding:**
Not only do I know how to use Hugging Face pipelines, but I also understand how models are loaded, tokenized, optimized, and deployed.
- **Experimentation:**
Tested a variety of cutting-edge models (Llama, Phi, Qwen, Starcoder, etc.) and documented differences in practical workflows.
- **Responsible Usage:**
Learned how to select and customize models, set device mappings, and handle model safety/inference warnings.


## How to Use

1. **Setup**
    - Open in Google Colab for GPU-accelerated experimentation.
    - Install necessary libraries (`transformers`, `diffusers`, `datasets`, `soundfile`, etc.)
2. **API Keys**
    - Set your Hugging Face and (optionally) OpenAI API keys using `google.colab.userdata`.
3. **Experiment**
    - Run section by section: try out sentiment analysis, question answering, translation, image generation, and more.
4. **Model Switching**
    - Easily modify code to swap between state-of-the-art models.
5. **Quantization Demos**
    - Check out the quantization sections to see real reduction in memory and speed-up using 4-bit weights.


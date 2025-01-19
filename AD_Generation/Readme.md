# 🌍 Multilingual Brochure Translator

## 📝 Project Overview
This project is a **Brochure Translation System** that translates a given English brochure into **Hindi** and **German** while maintaining its original value and appeal. The translation ensures that no information is lost or misinterpreted during conversion.

## 🚀 Features
- Translates an English brochure into **Hindi** and **German**.
- Ensures the translated text retains the **original intent and value**.
- **Streaming response** from OpenAI's API for real-time translation.
- Outputs the translation in a brochure-like format.

## 🛠️ Technologies Used
- **Python** 🐍
- **OpenAI API** (GPT-based translation)
- **Markdown Rendering** for formatted display

## 📌 How It Works
1. The system prompt (`translator_system_prompt`) defines the behavior of the assistant.
2. The function `get_translation_user_prompt(brochure)` prepares the structured prompt with the given brochure.
3. The function `translated_brochure()` streams responses from the OpenAI API and updates the Markdown display in real-time.

## 📚 Learnings & Challenges
### ✅ What I Learned:
- Using **OpenAI's Chat Completions API** for text translation.
- Implementing **real-time streaming responses** in Python.
- Ensuring translation quality while maintaining the **brochure’s format and appeal**.
- Handling **API responses efficiently** and displaying them dynamically.

### ⚡ Challenges Faced:
- Preserving the **context and structure** of the original brochure.
- Managing **API response formatting** and cleaning unnecessary Markdown tags.
- Optimizing the translation process for better performance.



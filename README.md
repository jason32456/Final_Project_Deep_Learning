# AI Text Summarizer - Deep Learning Final Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow)

## 📄 Overview

This repository contains the source code for the **AI Text Summarizer**, a Deep Learning Final Project for Semester 5. The application utilizes the **T5 (Text-To-Text Transfer Transformer)** model to perform abstractive text summarization, transforming lengthy news articles and documents into concise, meaningful summaries.

The project features an interactive web interface built with **Streamlit**, allowing users to input text, generate summaries in real-time, and analyze compression metrics.

## ✨ Key Features

* **Abstractive Summarization**: Generates new sentences to capture the essence of the text rather than just extracting existing lines.
* **Interactive UI**: User-friendly web interface for easy text input and visualization.
* **Performance Metrics**: Real-time calculation of compression ratios, word counts, and processing time.
* **Export Options**: Ability to download summaries as text files or copy them directly to the clipboard.
* **Sample Data**: Includes pre-loaded sample texts from CNN/Daily Mail for quick demonstration.
* **GPU Acceleration**: Automatically utilizes CUDA-enabled GPUs for faster inference if available.

## 🧠 Model & Data

### The Model
The application uses the `BrianAlex1/flan-t5-base-summarizer-news-model`, a fine-tuned version of Google's T5 model. T5 treats every NLP problem as a "text-to-text" task, making it highly effective for summarization.

* **Architecture**: T5 (Text-To-Text Transfer Transformer)
* **Task**: Abstractive Summarization
* **Framework**: PyTorch & Hugging Face Transformers

### The Dataset
The model works best with news-style articles, as reflected by the **CNN Daily Mail** dataset used in the context of this project.
* **Source**: CNN and Daily Mail news websites.
* **Size**: Over 300,000 news articles and summaries.
* **Style**: Journalistic/News format.

## 📂 Repository Structure

```text
├── app/
│   ├── app.py              # Main Streamlit application entry point
│   └── requirements.txt    # Dependencies specifically for running the app
├── data/
│   ├── README.md           # Documentation for the CNN Daily Mail dataset
│   └── cnn_dailymail/      # Dataset storage (excluded from git)
├── outputs/
│   └── model/              # Model artifacts (excluded from git)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Project-level dependencies
└── README.md               # Project documentation
```

### 🚀 Installation & Usage
Prerequisites
* **Python 3.8 or higher**
* **CUDA-capable GPU (optional, but recommended for faster performance)**

### Setup
* **Clone the repository**
```
git clone [https://github.com/jason32456/final_project_deep_learning.git](https://github.com/jason32456/final_project_deep_learning.git)
cd final_project_deep_learning
```
* **Install Dependencies It is recommended to use a virtual environment.**

```
# Install app-specific requirements
pip install -r app/requirements.txt
```

* **Run the Application Launch the Streamlit interface:**
```
streamlit run app/app.py
```

* **Access the App Open your browser and navigate to http://localhost:8501.**

### ⚙️ Configuration
The model runs with the following default generation parameters, which are optimized for news summarization:

* Max Length: 150 tokens

* Min Length: 30 tokens

* Num Beams: 4 (Beam Search)

* Length Penalty: 2.0

* No Repeat N-gram Size: 3

### 🛠️ Technology Stack
* Frontend: Streamlit

* Deep Learning: PyTorch, Hugging Face Transformers

* Data Processing: Numpy, Pandas

* Utilities: Textwrap, Pathlib

### 📝 Credits
Original Dataset: CNN & Daily Mail (See data/README.md for full details)

Base Model: Google T5 / Hugging Face Community

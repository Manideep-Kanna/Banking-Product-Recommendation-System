<div align="center">

# 🏦 Banking Product Recommendation System

### An AI-Powered Intelligent Recommendation Engine for Banking Products

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 📌 Overview

**Banking Product Recommendation System** is an AI-driven solution that analyzes customer profiles and recommends the most suitable banking products using **Retrieval-Augmented Generation (RAG)** and **LLM reasoning**. The system matches customer financial data against a curated product corpus to deliver personalized, explainable recommendations.

Key use cases:
- Personalized banking product recommendations (loans, credit cards, savings)
- Customer financial profile analysis
- Intelligent product-customer matching with LLM explanations
- Streamlit-based interactive demo interface

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────┐
│           Customer Profile Input                    │
│     (Age, Income, Credit Score, Goals...)          │
└─────────────────────┬─────────────────────┘
                       │
           ┌─────────┴─────────┐
           │   RAG Pipeline (LangChain)  │
           │  Query Embedding + FAISS   │
           └─────────┬─────────┘
                       │
           ┌─────────┴─────────┐
           │   Banking Product Store    │
           │  (Vector Embeddings/FAISS) │
           └─────────┬─────────┘
                       │
           ┌─────────┴─────────┐
           │  LLM Reasoning (GPT-4)    │
           │  Generate Recommendations │
           └─────────┬─────────┘
                       │
           Personalized Product Recommendations
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **RAG Pipeline** | Vector similarity search over banking product corpus for relevant retrieval |
| 🎯 **LLM Recommendations** | GPT-4 powered reasoning to match customer needs with products |
| 📊 **Customer Profiling** | Analyzes age, income, credit score, and financial goals |
| 🔍 **Semantic Search** | FAISS vector store for fast and accurate similarity matching |
| 💬 **Explainable AI** | LLM provides human-readable justification for each recommendation |
| 💾 **Persistent Store** | SQLite-backed customer and product data management |
| 🎨 **Interactive UI** | Streamlit-based web interface for demo and testing |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangChain |
| **LLM** | OpenAI GPT-4 / GPT-3.5-turbo |
| **Vector Store** | FAISS |
| **Embeddings** | OpenAI Embeddings |
| **UI** | Streamlit |
| **Database** | SQLite |
| **Language** | Python 3.8+ |

---

## ⚙️ Prerequisites

- Python 3.8+
- OpenAI API key

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Manideep-Kanna/Banking-Product-Recommendation-System.git
cd Banking-Product-Recommendation-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## 🏃 Usage

### Run the Application

```bash
streamlit run app.py
```

Or run the recommendation pipeline directly:

```bash
python first_phase.py
```

---

## 📁 Project Structure

```
Banking-Product-Recommendation-System/
├── app.py                        # Streamlit web application
├── first_phase.py                # Core recommendation pipeline
├── inhouse.db                    # SQLite database
├── unmatched_customers.txt       # Customers with no matching products
├── requirements.txt              # Python dependencies
├── .devcontainer/                # Dev Container configuration
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ by [Manideep Sitaram Kanna](https://github.com/Manideep-Kanna)

</div>

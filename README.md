# graduation-project
# SummARai: Arabic Book Summarization Platform


## 🌟 Introduction

SummARai (Summarizing Arabic with AI) is an innovative web-based platform designed to automatically generate high-quality summaries for Arabic books and documents using advanced AI technologies. As a graduation project from Cairo University's Faculty of Computers and Artificial Intelligence, SummARai addresses the critical gap in Arabic Natural Language Processing (NLP) tools by providing a specialized solution for Arabic content summarization.

## 🚀 Key Features

- **Hybrid Summarization Pipeline**: Combines extractive and abstractive techniques for optimal results
- **Arabic Language Specialization**: Specifically designed for Arabic's complex morphology and rich syntax
- **User-Friendly Interface**: Intuitive web platform with Arabic UI
- **Advanced AI Models**: Leverages AraT5 transformer and LLaMA 4 for superior summarization
- **Comprehensive Library**: Searchable database of book summaries
- **Reading Progress Tracking**: Personal statistics and reading history
- **OCR Support**: Processes both digital and scanned PDFs

## 🔧 Technology Stack

### Frontend
- React.js
- Tailwind CSS
- Axios for API communication

### Backend
- Java Spring Boot
- MySQL Database
- Redis Caching
- JWT Authentication

### Machine Learning
- Python FastAPI
- AraT5v2-base-1024 (fine-tuned)
- LLaMA-4-Scout-17B-16E-Instruct
- Mistral AI OCR
- Sentence Transformers for semantic chunking

### Infrastructure
- AWS Cloud Services
- Theta Labs GPU Resources (NVIDIA H100/A100)

## 📊 Performance Highlights

Our hybrid approach outperforms traditional methods:

| Metric          | Our Model | Baseline (mT5) |
|-----------------|-----------|----------------|
| BERTScore F1    | 71.73%    | 61.72%         |

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16+)
- Java JDK (v17+)
- Python (v3.9+)
- MySQL (v8.0+)
- Redis

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
./mvnw spring-boot:run
```

### ML Service Setup
```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📚 Dataset

We created SummARai v1.0 - a comprehensive dataset of 4,328 Arabic text chunks with aligned summaries, covering various genres. Available at:  
[SummARai Dataset GitHub](https://github.com/fatmaserry/SummARai_Dataset)

## 🤝 Team

- **Fatma Elzahraa Ashraf Samy**
- **Rowan Madeeh Mohamed**
- **Abdelrahman Mohamed Ezzat**
- **Abdelrahman Gomaa**
- **Momen Mostafa Mabrouk** 

**Supervised by**:  
Dr. Mohammed ElRamly  
TA. Amany Hesham

## 🌐 Live Demo

Explore SummARai: [Demo Link](https://shorturl.at/Ec0ds)


---

**SummARai** - Bridging the gap between Arabic literature and AI-powered accessibility.

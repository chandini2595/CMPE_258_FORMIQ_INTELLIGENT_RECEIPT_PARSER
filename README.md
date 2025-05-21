# FormIQ – Intelligent Receipt Parser & QA System (MLOps Project)

FormIQ is an intelligent, end-to-end receipt parsing system that combines OCR, instruction-tuned LLMs, and cloud infrastructure to extract, structure, and query receipt data. The solution automates the entire document understanding workflow—from image upload to structured storage and real-time querying—without requiring manually labeled bounding boxes.

Deployed on Hugging Face Spaces, FormIQ adheres to MLOps best practices by supporting modular retraining, result visualization, real-time chatbot inference, and streamlined CI/CD workflows.

## Project Information

- **Title**: FormIQ – Intelligent Receipt QA System
- **Course**: CMPE 258 - Deep Learning
- **Project Type**: MLOps Project
- **Team Members**:
  - Apurva Karne (018221801)
  - Chandini Saisri Uppuganti (018228483)
  - Manjunatha Inti (018192187)
  - Praful John (018168514)

## Abstract

Receipts are structurally diverse and often contain faded prints or handwritten annotations, making them difficult to digitize using traditional tools. FormIQ addresses this challenge through a modular pipeline that integrates PP-OCRv4 for robust text extraction, an instruction-tuned LLM via Perplexity API for semantic structuring, and Amazon DynamoDB for real-time, schema-less data storage.

The system delivers an end-to-end solution for receipt understanding without requiring any manually annotated bounding boxes. It achieves 92% end-to-end JSON validity, 84% accuracy on handwritten totals, and supports real-time chatbot querying with average response times around 1.3 seconds. A lightweight CNN module is also included for model training demonstration and visualization of key performance metrics. 

Deployed entirely on Hugging Face Spaces, FormIQ demonstrates production-ready deployment with MLOps practices including modular retraining, centralized inference, and evaluation monitoring.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Model and MLOps Practices](#model-and-mlops-practices)
- [Setup and Execution](#setup-and-execution)
- [Demo](#demo)
- [Team Contributions](#team-contributions)
- [Artifacts](#artifacts)
- [License](#license)
- [Notes for Evaluators](#notes-for-evaluators)

## Features

- Upload and process receipt or document images (PNG, JPG, JPEG, PDF)
- OCR-based text extraction using PP-OCRv4 (with Tesseract fallback)
- Structured information extraction using the Perplexity API
- Serverless storage of structured data in Amazon DynamoDB
- Natural language querying via FastAPI-powered chatbot using PartiQL
- Lightweight CNN training demo for model evaluation visualization
- Real-time metrics such as accuracy, loss, and confusion matrix
- Unified web interface built with Streamlit, deployed on Hugging Face Spaces

## Architecture

The architecture of FormIQ follows a modular and cloud-native design, enabling seamless integration of document parsing, structured data extraction, and interactive querying. The pipeline components are:

1. **Streamlit UI** – Handles document upload, result visualization, training demo, and chatbot access.
2. **OCR Engine (PP-OCRv4)** – Extracts raw text from uploaded images, including printed and handwritten content. Tesseract is used as a fallback.
3. **Perplexity API (LLM)** – Converts unstructured text into a normalized JSON schema with fields like vendor, date, items, and total.
4. **Amazon DynamoDB** – Stores the structured data in a serverless, scalable NoSQL table with real-time access.
5. **FastAPI Chatbot** – Accepts user queries, retrieves relevant data from DynamoDB, and returns contextual responses using PartiQL.
6. **CNN Training Module** – Provides real-time training and evaluation of a classification model with visualized metrics inside the UI.

![image](https://github.com/user-attachments/assets/1d03c31e-8a14-431b-9163-69e80e0899d7)

The entire system is deployed on Hugging Face Spaces, offering an accessible and reproducible MLOps demonstration.

## Technologies Used

| Component              | Technology                    |
|------------------------|-------------------------------|
| User Interface         | Streamlit                     |
| OCR Engine             | PP-OCRv4, Tesseract 5         |
| Language Model         | Perplexity API                |
| Backend API            | FastAPI                       |
| Database               | Amazon DynamoDB               |
| Query Language         | PartiQL                       |
| Model Training         | PyTorch (CNN demo)            |
| Data Processing        | Pandas, NumPy                 |
| Visualization          | Matplotlib, Seaborn, Plotly   |
| Deployment             | Hugging Face Spaces           |

## Model and MLOps Practices

FormIQ demonstrates key MLOps practices across the model lifecycle, with emphasis on modularity, observability, and deployment readiness:

- Model training is modularized through a lightweight CNN classifier, demonstrating supervised learning on synthetic data.
- Evaluation metrics such as accuracy, loss, and confusion matrix are visualized in real time within the Streamlit interface.
- Hyperparameters including optimizer (Adam), learning rate, batch size, and epochs are tracked and reported clearly.
- The Perplexity API integrates with the pipeline as a black-box LLM inference service, ensuring reproducibility via prompt logging and schema validation.
- FastAPI enables backend routing, clean API structure, and integration with Amazon DynamoDB and the chatbot.
- The system is deployed on Hugging Face Spaces to enable continuous access and reproducibility.
- The modular architecture allows for future integration with monitoring tools such as Evidently AI for drift detection and CI/CD pipelines for auto retraining and deployment.

## Setup and Execution

Follow the steps below to run FormIQ locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/formiq.git
   cd formiq
2. **Install required dependencies**
   ```
   pip install -r requirements.txt
3. **Set up environment variables**
   ```
   AWS_ACCESS_KEY_ID=YOUR_AWS_SECRET_ACCESS_KEY
   AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
   AWS_DEFAULT_REGION=YOUR_AWS_DEFAULT_REGION
   PERPLEXITY_API_KEY=YOUR_PERPLEXITY_API_KEY
   
4. **Run the application**
   ```
   streamlit run app.py
## MLOps Deployment

FormIQ is publicly deployed on Hugging Face Spaces.

You can access the live demo at the following URL:

[Access the Live App](https://huggingface.co/spaces/chandinisaisri/formiq)

The deployed version supports:
- Document upload and OCR processing
- JSON schema generation using Perplexity API
- Real-time chatbot interaction using FastAPI and DynamoDB
- Model training and evaluation visualization

## Team Contributions

| Team Member                | Student ID   | Contributions                                                                 |
|---------------------------|--------------|--------------------------------------------------------------------------------|
| Apurva Karne              | 018221801    | OCR integration, receipt dataset management, model evaluation module          |
| Chandini Saisri Uppuganti | 018228483    | Streamlit UI, Perplexity API integration, Hugging Face Spaces deployment      |
| Manjunatha Inti           | 018192187    | FastAPI backend development, DynamoDB storage integration, chatbot logic      |
| Praful John               | 018168514    | CNN training loop, evaluation visualization, documentation and error analysis |

## Artifacts

The following artifacts are included in this repository for evaluation and grading:

- [Slide Deck](https://www.slideshare.net/slideshow/form-iq-presentation-formiq-aims-to-eliminate-manual-receipt-entry-by-providing-an-end-to-end-single-page-web-application-that-automates-the-extraction-structuring-and-querying-of-receipt-data/279484322) – Final presentation slides
- [Watch our Project Demo - Powerpoint Presentation - Youtube]() - PPT Presentation
- [Detailed Project Explanation - Youtube]() - Project Walkthrough
- [Project Report](https://docs.google.com/document/d/1amdOJn9-E4yAmdxje4sl7iEWLuLOL6-eiTBXF-dvZtA/edit?usp=sharing)

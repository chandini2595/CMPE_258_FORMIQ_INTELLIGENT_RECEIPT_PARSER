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

## Artifacts

The following artifacts are included in this repository for evaluation and grading:

- [Slide Deck](https://www.slideshare.net/slideshow/form-iq-presentation-formiq-aims-to-eliminate-manual-receipt-entry-by-providing-an-end-to-end-single-page-web-application-that-automates-the-extraction-structuring-and-querying-of-receipt-data/279484322) – Final presentation slides
- [Watch our Project Demo - Powerpoint Presentation - Youtube](#) - PPT Presentation
- [Detailed Project Explanation - Youtube]() - Project Walkthrough
- [Project Report](https://docs.google.com/document/d/1amdOJn9-E4yAmdxje4sl7iEWLuLOL6-eiTBXF-dvZtA/edit?usp=sharing)

## Abstract

Receipts are structurally diverse and often contain faded prints or handwritten annotations, making them difficult to digitize using traditional tools. FormIQ addresses this challenge through a modular pipeline that integrates PP-OCRv4 for robust text extraction, an instruction-tuned LLM via Perplexity API for semantic structuring, and Amazon DynamoDB for real-time, schema-less data storage.

The system delivers an end-to-end solution for receipt understanding without requiring any manually annotated bounding boxes. It achieves 92% end-to-end JSON validity, 84% accuracy on handwritten totals, and supports real-time chatbot querying with average response times around 1.3 seconds. A lightweight CNN module is also included for model training demonstration and visualization of key performance metrics. 

Deployed entirely on Hugging Face Spaces, FormIQ demonstrates production-ready deployment with MLOps practices including modular retraining, centralized inference, and evaluation monitoring.

## Introduction

Receipt digitization presents unique challenges due to the variability in document formats, inclusion of handwritten annotations, and degradation of print quality over time. Traditional OCR tools often fail to produce semantically structured outputs, and layout-based deep learning models require large annotated datasets and complex training pipelines. 

FormIQ addresses these limitations through a lightweight, modular pipeline that integrates a state-of-the-art OCR engine (PP-OCRv4), an instruction-tuned large language model (Perplexity API), and serverless storage (Amazon DynamoDB) to automate the extraction and structuring of data from receipts and form-like documents. 

The project aims to provide a reproducible and cost-effective solution that can generalize across diverse layouts without requiring bounding box annotations or heavy infrastructure. Results show 84% success on handwritten totals, 92% valid structured outputs, and real-time chatbot response latency of approximately 1.3 seconds. The system is deployed using Hugging Face Spaces and adheres to MLOps best practices, including modular retraining, result visualization, and infrastructure-as-code readiness.

## Project UI

<img width="1510" alt="image" src="https://github.com/user-attachments/assets/b6e40289-9194-4b63-9be7-9b817f9bb23b" />

## Related Work

Traditional document processing systems often rely on rule-based OCR tools like Tesseract, which lack semantic understanding and struggle with handwritten or low-quality text. While layout-aware models such as LayoutLMv2 and LayoutLMv3 have demonstrated strong performance on structured document tasks, they typically require bounding box annotations and fine-tuning on large, domain-specific datasets—making them resource-intensive and less suitable for rapid prototyping.

FormIQ departs from these methods by employing a text-first, structure-later approach that decouples layout parsing from semantic understanding. This architecture allows the use of a powerful instruction-tuned LLM (via the Perplexity API) to semantically map raw OCR text to a strict JSON schema without annotated layout data.

Compared to commercial APIs like Amazon Textract or Google Document AI, which offer high accuracy but limited flexibility, FormIQ is open-source, modular, and cloud-native. It prioritizes reproducibility, cost-efficiency, and alignment with MLOps workflows over proprietary performance. Our system's unique combination of PaddleOCR, Perplexity API, and real-time FastAPI-based interaction positions it as a flexible alternative for academic and lightweight enterprise use cases.

## Data

FormIQ uses a combination of real and synthetic receipt images in PNG, JPG, and PDF formats. The dataset consists of approximately 300 receipts, covering a range of formats including printed, handwritten, faded, and annotated receipts. Real-world examples were sourced from public datasets such as SROIE and FUNSD, while synthetic data was generated using templated renderings and handwritten overlays to emulate realistic variations.

Each receipt contains fields such as vendor name, date, item list, total amount, and taxes. These documents are processed without bounding box annotations, as the pipeline uses a layout-agnostic strategy that relies on OCR text followed by LLM-based structuring.

All images are preprocessed using grayscale conversion, 300 DPI resampling, auto-rotation, and deskewing to optimize OCR quality. No explicit labeling was required due to the use of an instruction-aligned LLM (Perplexity API) for semantic mapping.

For training and evaluating the CNN classifier used in the demo module, a synthetic dataset of 600 document images was created, evenly distributed across three classes: Invoice, Misc Document, and Purchase Order. This dataset was used to demonstrate model behavior and visualize performance metrics within the Streamlit interface.

## Methods

FormIQ is built on a modular, cloud-native architecture that separates concerns across OCR extraction, LLM-based structuring, and interactive querying. This design enables flexibility, low maintenance overhead, and reproducibility—all aligned with MLOps best practices.

1. **OCR Extraction**: The pipeline begins with image preprocessing (grayscale, deskewing, 300 DPI resampling) to standardize inputs. Text is extracted using PP-OCRv4, which is optimized for multilingual and handwritten documents. Tesseract 5 serves as a fallback for high-quality printed text.

2. **Semantic Structuring via LLM**: The raw OCR output is passed to the Perplexity API, an instruction-tuned LLM that transforms unstructured text into a canonical JSON schema containing fields like vendor, items, date, and total. A retry mechanism ensures that non-conformant outputs are corrected through re-prompting.

3. **Data Storage**: Structured data is stored in Amazon DynamoDB, using a schema-flexible NoSQL model. This supports millisecond-scale retrieval and is optimized for PartiQL-based querying.

4. **Interactive Chatbot (FastAPI)**: The FastAPI backend serves as a bridge between the frontend UI and DynamoDB. It receives natural language queries, translates them into PartiQL, and returns formatted responses via a Streamlit-integrated chatbot.

5. **Model Training Module**: A CNN model was trained on a synthetic three-class dataset to demonstrate model evaluation. Key metrics (accuracy, loss, confusion matrix) are plotted in real-time within the UI, supporting transparency and experimentation.

6. **Deployment**: The full system is deployed on Hugging Face Spaces for easy access and reproducibility. The project structure supports integration with CI/CD, monitoring tools, and future auto-retraining pipelines.

Alternative approaches, such as layout-aware transformers (e.g., LayoutLMv3), were considered but not used due to their annotation requirements. The chosen approach balances performance, interpretability, and accessibility on a student budget.

## Experiments

We conducted a series of experiments to evaluate the performance, robustness, and reliability of each component in the FormIQ pipeline.

**OCR Performance**  
PP-OCRv4 and Tesseract 5 were benchmarked on 100 receipts (50 printed, 50 handwritten). PP-OCRv4 achieved 84% accuracy on handwritten receipts, while Tesseract reached only 54%. Average word recall was 90% with PP-OCRv4, demonstrating its effectiveness in low-contrast scenarios.

**LLM Structuring**  
Using the Perplexity API, 92% of structured outputs were valid JSON on the first attempt. A retry mechanism ensured 100% correction for malformed responses. Field coverage was 95%, confirming reliable extraction of vendor, date, items, and totals.

**Chatbot Responsiveness**  
Latency tests showed an average response time of 1.31 seconds per query. FastAPI routing and PartiQL execution maintained sub-second speeds, while LLM processing accounted for the bulk of the response time. This validates the system’s real-time interaction capability.

**CNN Classification**  
A CNN classifier trained on a 3-class synthetic dataset reached 82% accuracy and a macro F1-score of 0.93 after 10 epochs. The confusion matrix revealed that “Misc Doc” and “Invoice” were occasionally confused due to shared tabular structures. Training metrics were visualized live within the Streamlit interface.

**Ablation Studies**  
Removing key components showed measurable performance drops:
- Without preprocessing: OCR accuracy dropped by 12%
- Without grayscale normalization: 11% of totals were missed
- Without retry logic: JSON validity decreased by 8%

**Observation from Experiments**  
The experiments confirm that FormIQ delivers robust end-to-end document understanding, supported by careful preprocessing, semantic interpretation via LLMs, real-time query capabilities, and transparent model training visualizations.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Model and MLOps Practices](#model-and-mlops-practices)
- [Setup and Execution](#setup-and-execution)
- [Team Contributions](#team-contributions)

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

## Conclusion

FormIQ demonstrates that a lightweight, modular pipeline combining PP-OCRv4, an instruction-tuned LLM, and serverless cloud infrastructure can deliver production-grade receipt digitization without the need for bounding box annotations. The system achieved perfect performance on clean printed receipts, 84% success on handwritten totals, and 92% end-to-end structured JSON validity.

The chatbot interface, powered by FastAPI and DynamoDB, responds to natural language queries in an average of 1.3 seconds, validating its real-time capability. Additionally, the CNN model training module, though illustrative, achieved a macro F1-score of 0.93 and demonstrated the platform’s ability to visualize model metrics in real time.

FormIQ's design reflects MLOps best practices—modularization, reproducibility, and transparency—while remaining cost-effective and deployable on open platforms like Hugging Face Spaces. Future extensions include fine-tuning LayoutLMv3 for layout-aware extraction, adding real-time drift detection with Evidently AI, and supporting CI/CD workflows for automated retraining and deployment.

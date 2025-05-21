# FormIQ Intelligent Receipt Parser - Supplementary Material

## 1. Technical Architecture

### 1.1 Core Components
- **LayoutLMv3 Integration**: Advanced document understanding model for layout analysis
- **PaddleOCR**: Handwritten text recognition
- **Perplexity AI**: LLM-based information extraction
- **Streamlit**: Interactive web interface
- **DynamoDB**: Cloud storage for receipt data

### 1.2 Key Technical Features

#### Document Processing Pipeline
```python
# Core document processing flow
1. Image Upload → Preprocessing
2. OCR Processing (PaddleOCR)
3. Layout Analysis (LayoutLMv3)
4. Information Extraction (Perplexity AI)
5. Data Validation & Storage
```

#### Advanced Text Extraction
```python
def extract_with_perplexity_llm(ocr_text):
    # Uses Perplexity AI for intelligent information extraction
    # Handles complex receipt formats and variations
    # Returns structured JSON data
```

#### Handwritten Text Recognition
```python
def extract_handwritten_text_paddle(image):
    # Uses PaddleOCR for handwritten text recognition
    # Supports multiple languages and orientations
    # Real-time processing capabilities
```

## 2. Technical Implementation Details

### 2.1 Model Integration
- LayoutLMv3 for document layout understanding
- PaddleOCR for handwritten text recognition
- Perplexity AI for intelligent information extraction

### 2.2 Data Processing
- Image preprocessing and enhancement
- OCR text extraction
- Structured data parsing
- Validation and error handling

### 2.3 Cloud Integration
- DynamoDB for scalable storage
- Secure data handling
- Real-time processing capabilities

## 3. Performance Metrics

### 3.1 Accuracy Metrics
- OCR accuracy rates
- Information extraction precision
- Handwritten text recognition success rates

### 3.2 Processing Speed
- Average processing time per document
- Real-time processing capabilities
- Scalability metrics

## 4. Technical Requirements

### 4.1 Dependencies
```txt
- Python 3.8+
- PyTorch
- Transformers
- PaddleOCR
- Streamlit
- AWS SDK (boto3)
- OpenAI/Perplexity API
```

### 4.2 System Requirements
- GPU support for model inference
- Minimum 4GB RAM
- Internet connection for API access

## 5. Future Technical Enhancements

### 5.1 Planned Improvements
1. Multi-document batch processing
2. Enhanced error correction
3. Custom model fine-tuning
4. Additional language support
5. Real-time collaboration features

### 5.2 Technical Roadmap
- Integration with additional OCR engines
- Enhanced validation rules
- Improved error handling
- Performance optimization
- Extended API capabilities

# FormIQ Intelligent Receipt Parser - Supplementary Material

## 1. Technical Architecture

### 1.1 Core Components
- **LayoutLMv3 Integration**: Advanced document understanding model for layout analysis
- **PaddleOCR**: Handwritten text recognition
- **Perplexity AI**: LLM-based information extraction
- **Streamlit**: Interactive web interface
- **DynamoDB**: Cloud storage for receipt data

### 1.2 Key Technical Features

#### Document Processing Pipeline
```python
# Core document processing flow
1. Image Upload → Preprocessing
2. OCR Processing (PaddleOCR)
3. Layout Analysis (LayoutLMv3)
4. Information Extraction (Perplexity AI)
5. Data Validation & Storage
```

#### Advanced Text Extraction
```python
def extract_with_perplexity_llm(ocr_text):
    # Uses Perplexity AI for intelligent information extraction
    # Handles complex receipt formats and variations
    # Returns structured JSON data
```

#### Handwritten Text Recognition
```python
def extract_handwritten_text_paddle(image):
    # Uses PaddleOCR for handwritten text recognition
    # Supports multiple languages and orientations
    # Real-time processing capabilities
```

## 2. Technical Implementation Details

### 2.1 Model Integration
- LayoutLMv3 for document layout understanding
- PaddleOCR for handwritten text recognition
- Perplexity AI for intelligent information extraction

### 2.2 Data Processing
- Image preprocessing and enhancement
- OCR text extraction
- Structured data parsing
- Validation and error handling

### 2.3 Cloud Integration
- DynamoDB for scalable storage
- Secure data handling
- Real-time processing capabilities

## 3. Performance Metrics

### 3.1 Accuracy Metrics
- OCR accuracy rates
- Information extraction precision
- Handwritten text recognition success rates

### 3.2 Processing Speed
- Average processing time per document
- Real-time processing capabilities
- Scalability metrics

## 4. Technical Requirements

### 4.1 Dependencies
```txt
- Python 3.8+
- PyTorch
- Transformers
- PaddleOCR
- Streamlit
- AWS SDK (boto3)
- OpenAI/Perplexity API
```

### 4.2 System Requirements
- GPU support for model inference
- Minimum 4GB RAM
- Internet connection for API access

## 5. Future Technical Enhancements

### 5.1 Planned Improvements
1. Multi-document batch processing
2. Enhanced error correction
3. Custom model fine-tuning
4. Additional language support
5. Real-time collaboration features

### 5.2 Technical Roadmap
- Integration with additional OCR engines
- Enhanced validation rules
- Improved error handling
- Performance optimization
- Extended API capabilities 

## 6. Project Structure and Organization

### 6.1 Source Code Organization
```
FormIQ/
├── src/
│   ├── frontend/
│   │   └── app.py                 # Streamlit UI application
│   ├── api/
│   │   └── main.py               # API controller for backend logic
│   ├── scripts/
│   │   └── train.py              # CNN training script with evaluation visualization
│   ├── models/
│   │   └── layoutlm.py           # LayoutLMv3 architecture placeholder
│   └── tests/
│       └── test_model.py         # Unit test file for model pipeline
├── chatbot_server.py             # FastAPI server for chatbot
├── chatbot_utils.py              # Helper utilities for chatbot logic
```

### 6.2 Configuration Files
```
FormIQ/
├── config/
│   └── config.yaml               # Centralized configuration
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
└── packages.txt                  # OS-level dependencies
```

### 6.3 Deployment and CI/CD
```
FormIQ/
├── Dockerfile                    # Container definition
└── .github/
    └── workflows/               # GitHub Actions CI/CD workflows
```

### 6.4 Documentation and Assets
```
FormIQ/
├── README.md                     # Project documentation
├── SUPPLEMENTARY_MATERIAL.md     # Technical reference guide
├── temp_uploaded_image.jpg       # Sample receipt image
└── temp_uploaded_image_paddle.jpg # PaddleOCR test image
```

### 6.5 Key Components Description

#### Frontend Application (src/frontend/app.py)
- Streamlit-based user interface
- Real-time document processing
- Interactive visualization of results
- Error handling and user feedback

#### Backend API (src/api/main.py)
- RESTful API endpoints
- Document processing pipeline
- Integration with ML models
- Data validation and storage

#### Model Training (src/scripts/train.py)
- CNN model training pipeline
- Evaluation metrics visualization
- Model checkpointing
- Performance monitoring

#### LayoutLM Integration (src/models/layoutlm.py)
- LayoutLMv3 model architecture
- Document understanding pipeline
- Custom model extensions
- Inference optimization

#### Testing Framework (src/tests/test_model.py)
- Unit tests for model pipeline
- Integration tests
- Performance benchmarks
- Error case handling

#### Chatbot Integration
- FastAPI server implementation
- Natural language processing
- Context-aware responses
- Error handling and recovery

### 6.6 Configuration Management
- Environment-specific settings
- API key management
- Model parameters
- Pipeline configurations

### 6.7 Deployment Infrastructure
- Docker containerization
- CI/CD pipeline automation
- Environment consistency
- Scalability considerations

### 6.8 Visual Assets and Documentation
- Sample receipt images
- UI screenshots
- Processing flow diagrams
- Technical documentation

### 6.9 Dependencies Management
- Python package requirements
- System-level dependencies
- Version control
- Compatibility matrix 

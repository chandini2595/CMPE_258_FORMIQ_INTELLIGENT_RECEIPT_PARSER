---
title: FormIQ - Intelligent Document Parser
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# FormIQ - Intelligent Document Parser

FormIQ is an intelligent document parser that uses advanced AI models to extract and validate information from various types of documents.

## Features

- Document image upload and processing
- OCR text extraction using Tesseract
- Advanced document understanding using LayoutLMv3
- Structured information extraction using Perplexity AI
- Interactive web interface built with Streamlit

## Technologies Used

- **Frontend**: Streamlit
- **OCR**: Tesseract
- **Document Understanding**: LayoutLMv3
- **Text Processing**: Perplexity AI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   PERPLEXITY_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the provided URL
3. Upload a document image
4. Click "Process Document" to extract information

## Hugging Face Spaces Deployment

This project is deployed on Hugging Face Spaces. You can access the live demo at: [Your Spaces URL]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

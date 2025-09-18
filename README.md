# Mistral OCR Component for FRA Document Processing

This component uses Mistral's OCR API to extract coordinates and location details from FRA (Forest Rights Act) documents and converts them to GeoJSON format.

## Features

- **Mistral OCR Integration**: Uses Mistral's advanced OCR API for text extraction
- **Coordinate Extraction**: Identifies lat/long coordinates, survey numbers, and plot IDs
- **GeoJSON Conversion**: Converts extracted data to GeoJSON format
- **Batch Processing**: Supports bulk document processing using Mistral Batch API
- **Multiple Formats**: Handles PDFs, images (JPG, PNG, TIFF)

## Setup

1. Copy .env.example to .env and configure your settings
2. Install dependencies: pip install -r requirements.txt
3. Set up PostgreSQL with PostGIS extension
4. Get your Mistral API key from [Mistral Console](https://console.mistral.ai/api-keys/)
5. Start the application: uvicorn app.main:app --reload

## API Endpoints

- POST /upload - Upload document for OCR processing
- GET /status/{task_id} - Check processing status
- GET /result/{task_id} - Get GeoJSON result
- POST /batch - Batch process multiple documents

## Usage Example

`python
import requests

# Upload a document
files = {'file': open('fra_document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)
task_id = response.json()['task_id']

# Check status
status = requests.get(f'http://localhost:8000/status/{task_id}')

# Get result
result = requests.get(f'http://localhost:8000/result/{task_id}')
geojson_data = result.json()
`

## Technology Stack

- **FastAPI**: Web framework
- **Mistral OCR**: Text extraction from documents
- **PostgreSQL + PostGIS**: GeoJSON storage
- **Redis**: Task queue and caching
- **Celery**: Background task processing

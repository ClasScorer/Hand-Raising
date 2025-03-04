# Hand Raising Detection API

A machine learning API that detects hand gestures in images, with a focus on identifying raised hands and hand-face interactions. Built with FastAPI, MediaPipe, and OpenCV.

## Features

- **Hand Raise Detection**: Identifies when a hand is raised above wrist level in an image
- **Face-Touch Detection**: Detects when hands are touching or near the face
- **Multiple Hand Support**: Can detect and analyze up to 5 hands simultaneously
- **Gesture Analysis**: Advanced landmark detection using MediaPipe
- **Rate Limiting**: Built-in protection against API abuse
- **Comprehensive Validation**: Robust input validation and error handling
- **Docker Support**: Easy deployment with containerization

## API Endpoints

### POST `/detect-hand`

Analyzes an uploaded image to detect hand gestures.

**Responses:**
- `Hand Raised`: At least one hand is raised above wrist level
- `Hand Touching Face`: At least one hand is detected touching or near the face
- `No Hand Raised`: Hands are detected but not raised
- `No Hand Detected`: No hands were found in the image

### GET `/`

Health check endpoint to verify API status and version.

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- FastAPI
- Uvicorn
- Docker (optional for containerized deployment)

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Hand-Raising
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API:
   ```bash
   uvicorn app:app --reload
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t hand-detection-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 hand-detection-api
   ```

## Usage Examples

### Python Client

```python
import requests

url = "http://localhost:8000/detect-hand"
image_path = "path/to/your/image.jpg"

with open(image_path, "rb") as image_file:
    files = {"file": (image_path, image_file, "image/jpeg")}
    response = requests.post(url, files=files)

result = response.json()
print(f"Detection result: {result['result']}")
print(f"Details: {result['details']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/detect-hand" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg;type=image/jpeg"
```

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- ReDoc alternative documentation: http://localhost:8000/redoc

## Configuration

The API can be configured using environment variables:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| APP_DEBUG | Enable debug mode | False |
| APP_MAX_IMAGE_SIZE | Maximum file size in bytes | 10485760 (10MB) |
| APP_MIN_DETECTION_CONFIDENCE | MediaPipe detection threshold | 0.7 |
| APP_ALLOWED_ORIGINS | CORS allowed origins | * |

## Limitations

- Maximum file size: 10MB
- Supported image formats: JPEG, PNG
- Rate limit: 10 requests per minute per IP

## License

[MIT License](LICENSE)

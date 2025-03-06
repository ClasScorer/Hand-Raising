# Hand Raising Detection API

A FastAPI-based application that detects hand raising gestures in images using MediaPipe and stores the results in a PostgreSQL database using Prisma ORM.

## Features

- Hand raise detection using MediaPipe
- Face interaction detection
- Multiple hand support (up to 5 hands)
- Database storage of detection results
- Rate limiting and request tracking
- Comprehensive error handling and logging
- Swagger UI documentation

## Prerequisites

- Python 3.8+
- PostgreSQL 13+
- Docker and Docker Compose
- Virtual Environment (venv)

## Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Hand-Raising
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with the following variables:
```env
# Database Configuration
DATABASE_URL="postgresql://username:password@localhost:5432/hand_raising_db"

# Application Settings
APP_DEBUG=true
APP_MAX_IMAGE_SIZE=10485760  # 10MB in bytes
APP_MIN_DETECTION_CONFIDENCE=0.7
APP_MIN_TRACKING_CONFIDENCE=0.5
APP_MAX_NUM_HANDS=5
APP_MAX_NUM_FACES=1
APP_HAND_FACE_DISTANCE_THRESHOLD=0.1
```

## Database Setup

1. Start the PostgreSQL container:
```bash
docker compose up -d
```

2. Generate Prisma client and push the schema:
```bash
python -m prisma generate
python -m prisma db push
```

## Running the Application

1. Ensure the virtual environment is activated:
```bash
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Start the FastAPI server:
```bash
uvicorn app:app --reload --log-level debug
```

The API will be available at `http://localhost:23122`

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:23122/docs`
- ReDoc documentation: `http://localhost:23122/redoc`

## API Endpoints

### 1. Health Check
```
GET /
```
Verifies if the API is running and returns version information.

### 2. Hand Detection
```
POST /detect-hand
```
Detects hand gestures in an uploaded image without storing results.

### 3. Hand Raising Detection and Storage
```
POST /detect-hand-raising
```
Detects hand raising and stores results in the database.

Required form fields:
- `file`: Image file (JPEG/PNG)
- `student_id`: Student identifier
- `lecture_id`: Lecture identifier
- `timestamp`: ISO format timestamp (e.g., "2024-03-21T14:30:00")

## Response Format

### Success Response
```json
{
    "student_id": "student123",
    "timestamp": "2024-03-21T14:30:00",
    "status": "RAISED",
    "confidence": 0.95,
    "hand_position": {
        "x": 0.5,
        "y": 0.3,
        "z": 0.1
    }
}
```

### Hand Raising Status Values
- `RAISED`: Hand is raised above the wrist level
- `NOT_RAISED`: Hand is detected but not raised
- `ON_FACE`: Hand is detected near or touching the face

## Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- File size limits (max 10MB)
- Invalid timestamp formats
- Database connection issues
- Rate limiting (10 requests per minute)

## Development

### Logging
- Debug logs are enabled when `APP_DEBUG=true`
- Logs include request IDs, processing times, and detailed error information
- All logs are output to stdout

### Database Schema
The application uses Prisma ORM with the following schema:
```prisma
model HandRaisingSchema {
    studentId  String
    lectureId  String
    timestamp  DateTime
    status     HandRaisingStatus
    confidence Float
    positionX  Float?
    positionY  Float?
    positionZ  Float?

    @@id([studentId, lectureId, timestamp])
}
```

## Security Considerations

1. Environment Variables:
   - Never commit `.env` file to version control
   - Use secure credentials in production
   - Add `.env` to `.gitignore`

2. Rate Limiting:
   - Configured to 10 requests per minute per IP
   - Can be adjusted in the code if needed

3. File Upload Security:
   - File size limited to 10MB
   - Only image files (JPEG/PNG) are accepted
   - File content validation before processing

## Troubleshooting

1. Database Connection Issues:
   - Verify PostgreSQL container is running
   - Check DATABASE_URL in .env
   - Ensure database exists and is accessible

2. Image Processing Errors:
   - Verify image format (JPEG/PNG only)
   - Check image is not corrupted
   - Ensure image size is within limits

3. Hand Detection Issues:
   - Ensure good lighting in images
   - Check if hands are clearly visible
   - Verify confidence thresholds in .env

## License

[Add your license information here]

import cv2
import mediapipe as mp # type: ignore
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, BaseSettings, Field
from enum import Enum
import logging
from typing import List, Optional, Set, Dict
import sys
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from datetime import datetime
from prisma import Prisma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize Prisma client with connection management
prisma = Prisma()

@app.on_event("startup")
async def startup():
    """Connect to the database when the app starts"""
    try:
        await prisma.connect()
        logger.info("Successfully connected to the database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        # Re-raise the exception to prevent the app from starting with a broken DB connection
        raise

@app.on_event("shutdown")
async def shutdown():
    """Disconnect from the database when the app shuts down"""
    try:
        await prisma.disconnect()
        logger.info("Successfully disconnected from the database")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {str(e)}")
        raise

# Configuration management
class Settings(BaseSettings):
    app_name: str = "Hand Detection API"
    version: str = "1.0.0"
    debug: bool = os.getenv('APP_DEBUG', 'false').lower() == 'true'
    allowed_origins: Set[str] = {"*"}
    max_image_size: int = int(os.getenv('APP_MAX_IMAGE_SIZE', 10 * 1024 * 1024))
    min_detection_confidence: float = float(os.getenv('APP_MIN_DETECTION_CONFIDENCE', 0.7))
    min_tracking_confidence: float = float(os.getenv('APP_MIN_TRACKING_CONFIDENCE', 0.5))
    max_num_hands: int = int(os.getenv('APP_MAX_NUM_HANDS', 5))
    max_num_faces: int = int(os.getenv('APP_MAX_NUM_FACES', 1))
    hand_face_distance_threshold: float = float(os.getenv('APP_HAND_FACE_DISTANCE_THRESHOLD', 0.1))

    class Config:
        env_prefix = "APP_"

settings = Settings()
limiter = Limiter(key_func=get_remote_address)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Enhanced response models with examples
class HandRaisingStatus(str, Enum):
    """
    Possible states of hand raising detection.
    
    Values:
        RAISED: Hand is raised above the wrist level
        NOT_RAISED: Hand is detected but not raised
        ON_FACE: Hand is detected near or touching the face
    """
    RAISED = "RAISED"
    NOT_RAISED = "NOT_RAISED"
    ON_FACE = "ON_FACE"

class DetectionResponse(BaseModel):
    """
    Response model for hand detection results.
    """
    status: str = Field(
        description="Status of the request processing",
        example="success"
    )
    result: HandRaisingStatus = Field(
        description="The detected hand raising state"
    )
    details: List[str] = Field(
        description="Additional details about the detection",
        example=["Hands detected: 1"]
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "result": "RAISED",
                "details": ["Hands detected: 1"]
            }
        }

class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """
    status: str = Field(
        description="Error status",
        example="error"
    )
    message: str = Field(
        description="Detailed error message",
        example="File size exceeds maximum allowed size of 10MB"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "File must be an image"
            }
        }

class HandRaisingResponse(BaseModel):
    student_id: str
    timestamp: datetime
    status: HandRaisingStatus
    confidence: float
    hand_position: Optional[Dict[str, float]] = None

# API Documentation
API_DESCRIPTION = """
# Hand Detection API

This API provides endpoints for detecting hand gestures and face interactions in images.

## Features

- Hand raise detection
- Hand-face interaction detection
- Multiple hand detection support
- Face mesh detection

## Usage Limits

- Maximum file size: 10MB
- Rate limit: 10 requests per minute per IP
- Supported image formats: JPEG, PNG
- Maximum number of hands detected: 5
- Maximum number of faces detected: 1

## Error Codes

- 400: Bad Request (Invalid input)
  - Invalid file type
  - Invalid image format
  - Corrupted image data
- 413: Payload Too Large
  - File size exceeds 10MB limit
- 429: Too Many Requests
  - Rate limit exceeded
- 500: Internal Server Error
  - Processing error
  - System error

## Headers

Response headers include:
- `X-Process-Time`: Processing time in seconds
- `X-Request-ID`: Unique request identifier
- `X-RateLimit-Limit`: Rate limit per minute
- `X-RateLimit-Remaining`: Remaining requests in current window
"""

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title=settings.app_name,
    description=API_DESCRIPTION,
    version=settings.version,
    debug=settings.debug,
    openapi_tags=[
        {
            "name": "Detection",
            "description": "Endpoints for hand gesture and face interaction detection"
        },
        {
            "name": "Health",
            "description": "Endpoints for API health monitoring"
        }
    ],
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe components
try:
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=settings.max_num_hands,
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence
    )
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=settings.max_num_faces,
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence
    )
except Exception as e:
    logger.error(f"Failed to initialize Mediapipe components: {str(e)}")
    raise

def is_hand_touching_face(hand_landmarks, face_landmarks, threshold: float = settings.hand_face_distance_threshold) -> bool:
    """
    Check if any hand landmarks are near the face landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        face_landmarks: MediaPipe face landmarks
        threshold: Distance threshold for considering hand-face interaction
        
    Returns:
        bool: True if hand is touching face, False otherwise
    """
    try:
        hand_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        face_coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

        hand_array = np.array(hand_coords)
        face_array = np.array(face_coords)

        for hand_point in hand_array:
            distances = np.linalg.norm(face_array - hand_point, axis=1)
            if np.any(distances < threshold):
                return True
        return False
    except Exception as e:
        logger.error(f"Error in hand-face detection: {str(e)}")
        return False

def is_hand_raised(hand_landmarks) -> bool:
    """
    Check if the hand is raised.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        bool: True if hand is raised, False otherwise
    """
    try:
        wrist_y = hand_landmarks.landmark[0].y
        thumb_tip_y = hand_landmarks.landmark[4].y
        index_tip_y = hand_landmarks.landmark[8].y
        
        return thumb_tip_y < wrist_y and index_tip_y < wrist_y
    except Exception as e:
        logger.error(f"Error in hand raise detection: {str(e)}")
        return False

async def process_image(image_data) -> DetectionResponse:
    """
    Process the image and detect hand gestures.
    
    Args:
        image_data: OpenCV image data
        
    Returns:
        DetectionResponse: Detection results
    """
    try:
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        face_results = face_mesh.process(rgb_image)

        response = DetectionResponse(
            status="success",
            result=HandRaisingStatus.NOT_RAISED,  # Default to not raised
            details=[]
        )

        if not results.multi_hand_landmarks:
            response.details.append("No hands detected")
            return response

        num_hands = len(results.multi_hand_landmarks)
        response.details.append(f"Hands detected: {num_hands}")
        logger.info(f"Processing image with {num_hands} hands detected")

        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

        for hand_landmarks in results.multi_hand_landmarks:
            if face_landmarks and is_hand_touching_face(hand_landmarks, face_landmarks):
                response.result = HandRaisingStatus.ON_FACE
                break
            
            if is_hand_raised(hand_landmarks):
                response.result = HandRaisingStatus.RAISED
                break

        return response

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class RequestValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add request ID and timestamp
        request.state.request_id = f"{int(time.time())}_{id(request)}"
        request.state.start_time = datetime.utcnow()
        
        # Log incoming request
        logger.info(
            f"Request started - ID: {request.state.request_id} "
            f"Method: {request.method} Path: {request.url.path}"
        )
        
        response = await call_next(request)
        
        # Calculate and add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.state.request_id
        
        # Log response
        logger.info(
            f"Request completed - ID: {request.state.request_id} "
            f"Status: {response.status_code} "
            f"Processing Time: {process_time:.3f}s"
        )
        
        return response

# Add middleware to app
app.add_middleware(RequestValidationMiddleware)

@app.post("/detect-hand", 
         response_model=DetectionResponse,
         responses={
             400: {
                 "model": ErrorResponse,
                 "description": "Invalid input",
                 "content": {
                     "application/json": {
                         "examples": {
                             "invalid_type": {
                                 "summary": "Invalid file type",
                                 "value": {"status": "error", "message": "File must be an image"}
                             },
                             "invalid_format": {
                                 "summary": "Invalid image format",
                                 "value": {"status": "error", "message": "Invalid image file or format"}
                             }
                         }
                     }
                 }
             },
             413: {
                 "model": ErrorResponse,
                 "description": "File too large",
                 "content": {
                     "application/json": {
                         "example": {"status": "error", "message": "File size exceeds maximum allowed size of 10MB"}
                     }
                 }
             },
             429: {
                 "model": ErrorResponse,
                 "description": "Rate limit exceeded",
                 "content": {
                     "application/json": {
                         "example": {"status": "error", "message": "Rate limit exceeded: 10 requests per minute"}
                     }
                 }
             },
             500: {
                 "model": ErrorResponse,
                 "description": "Internal server error",
                 "content": {
                     "application/json": {
                         "example": {"status": "error", "message": "Internal server error during image processing"}
                     }
                 }
             }
         },
         tags=["Detection"],
         summary="Detect hand gestures in an image",
         description="""
Upload an image to detect hand raising and face touching gestures.

**Input Image Requirements:**
- Format: JPEG, PNG
- Maximum size: 10MB
- Minimum resolution: 64x64
- Maximum resolution: 4096x4096
- Color space: RGB or BGR

**Detection Capabilities:**
- Hand raise detection: Detects if hands are raised above wrist level
- Face touch detection: Detects if hands are near or touching the face
- Multiple hand support: Can detect up to 5 hands
- Face detection: Supports single face detection

**Example Usage:**
```python
import requests

url = "http://localhost:8000/detect-hand"
files = {"file": ("image.jpg", open("image.jpg", "rb"), "image/jpeg")}
response = requests.post(url, files=files)
print(response.json())
```

**Example Response:**
```json
{
    "status": "success",
    "result": "RAISED",
    "details": ["Hands detected: 1"]
}
```
         """)
@limiter.limit("10/minute")
async def detect_hand(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    """
    Endpoint to receive an image and return hand detection results.
    
    Args:
        file: Uploaded image file
        
    Returns:
        DetectionResponse: Detection results
        
    Raises:
        HTTPException: For invalid input or processing errors
    """
    try:
        logger.info(f"Processing image upload - Request ID: {request.state.request_id}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type} - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Validate file size
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"File size: {file_size/1024:.2f}KB - Request ID: {request.state.request_id}")
        
        if file_size > settings.max_image_size:
            logger.warning(f"File too large: {file_size/1024/1024:.2f}MB - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {settings.max_image_size // (1024 * 1024)}MB"
            )

        # Process image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Failed to decode image - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or format"
            )
        
        result = await process_image(image)
        logger.info(f"Detection result: {result.result} - Request ID: {request.state.request_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)} - Request ID: {request.state.request_id}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )

@app.get("/",
         response_model=dict,
         responses={
             200: {
                 "description": "Successful response",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "ok",
                             "message": "Hand detection API is running",
                             "version": "1.0.0"
                         }
                     }
                 }
             }
         },
         tags=["Health"],
         summary="Health check endpoint",
         description="""
Verify if the API is running and get version information.

This endpoint can be used to:
- Check if the API is operational
- Get the current API version
- Verify connectivity

No authentication required.
         """)
async def root() -> dict:
    """Root endpoint to verify the API is running."""
    return {
        "status": "ok",
        "message": "Hand detection API is running",
        "version": settings.version
    }

def detect_hand_raising_in_image(image) -> tuple[HandRaisingStatus, float, dict]:
    """
    Detect if a hand is raised in the image.
    
    Args:
        image: OpenCV image data
        
    Returns:
        tuple: (status, confidence, hand_position)
            - status (HandRaisingStatus): The detected hand raising status
            - confidence (float): Confidence score of the detection
            - hand_position (dict): Position data of the detected hand
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        # Default return values
        status = HandRaisingStatus.NOT_RAISED
        confidence = 0.0
        hand_position = {}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_handedness = results.multi_handedness[0]
            
            # Get hand position data
            wrist = hand_landmarks.landmark[0]  # Wrist landmark
            hand_position = {
                'x': wrist.x,
                'y': wrist.y,
                'z': wrist.z
            }
            
            # Calculate confidence based on landmark detection scores
            confidence = float(hand_handedness.classification[0].score)
            
            # Check if hand is touching face
            face_results = face_mesh.process(rgb_image)
            face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
            
            if face_landmarks and is_hand_touching_face(hand_landmarks, face_landmarks):
                status = HandRaisingStatus.ON_FACE
            # Calculate if hand is raised using existing logic
            elif is_hand_raised(hand_landmarks):
                status = HandRaisingStatus.RAISED
            
            logger.info(f"Hand detection: status={status}, confidence={confidence:.2f}")
        else:
            logger.info("No hands detected in the image")
            
        return status, confidence, hand_position
        
    except Exception as e:
        logger.error(f"Error in hand raising detection: {str(e)}")
        raise

@app.post("/detect-hand-raising", response_model=HandRaisingResponse)
async def detect_hand_raising(
    request: Request,
    file: UploadFile = File(...),
    student_id: str = Form(...),
    timestamp: str = Form(...),
    lecture_id: str = Form(...)
):
    try:
        logger.info(f"Processing hand raising detection - Request ID: {request.state.request_id}")
        logger.debug(f"Parameters - student_id: {student_id}, lecture_id: {lecture_id}, timestamp: {timestamp}")
        
        # Validate timestamp format
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as e:
            logger.warning(f"Invalid timestamp format: {timestamp} - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timestamp format. Expected ISO format (e.g., 2024-03-21T14:30:00). Error: {str(e)}"
            )

        # Process the image
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Failed to decode image - Request ID: {request.state.request_id}")
                raise HTTPException(status_code=400, detail="Invalid image file or format")
                
            logger.info(f"Successfully decoded image - Request ID: {request.state.request_id}")
        except Exception as img_error:
            logger.error(f"Error processing image: {str(img_error)} - Request ID: {request.state.request_id}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(img_error)}")

        # Detect hand raising
        try:
            status, confidence, hand_position = detect_hand_raising_in_image(image)
            logger.info(f"Hand detection results - status: {status}, confidence: {confidence:.2f} - Request ID: {request.state.request_id}")
        except Exception as detect_error:
            logger.error(f"Error in hand detection: {str(detect_error)} - Request ID: {request.state.request_id}")
            raise HTTPException(status_code=500, detail=f"Error in hand detection: {str(detect_error)}")

        # Push to database using Prisma
        try:
            # Ensure database connection
            if not prisma.is_connected():
                logger.warning("Prisma client not connected, attempting to reconnect...")
                await prisma.connect()
                
            await prisma.handraisingschema.create(
                data={
                    'studentId': student_id,
                    'lectureId': lecture_id,
                    'timestamp': parsed_timestamp,
                    'status': status,
                    'confidence': confidence,
                    'positionX': hand_position.get('x'),
                    'positionY': hand_position.get('y'),
                    'positionZ': hand_position.get('z')
                }
            )
            logger.info(f"Successfully saved to database - Request ID: {request.state.request_id}")
            
            return HandRaisingResponse(
                student_id=student_id,
                timestamp=parsed_timestamp,
                status=status,
                confidence=confidence,
                hand_position=hand_position if hand_position else None
            )
            
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)} - Request ID: {request.state.request_id}")
            # Check if it's a connection error and try to reconnect
            if "not connected" in str(db_error).lower():
                try:
                    await prisma.connect()
                    # Retry the operation
                    await prisma.handraisingschema.create(
                        data={
                            'studentId': student_id,
                            'lectureId': lecture_id,
                            'timestamp': parsed_timestamp,
                            'status': status,
                            'confidence': confidence,
                            'positionX': hand_position.get('x'),
                            'positionY': hand_position.get('y'),
                            'positionZ': hand_position.get('z')
                        }
                    )
                    logger.info("Successfully saved to database after reconnection")
                    return HandRaisingResponse(
                        student_id=student_id,
                        timestamp=parsed_timestamp,
                        status=status,
                        confidence=confidence,
                        hand_position=hand_position if hand_position else None
                    )
                except Exception as retry_error:
                    logger.error(f"Failed to reconnect and save to database: {str(retry_error)}")
                    raise HTTPException(
                        status_code=500,
                        detail="Database connection error. Please try again later."
                    )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save hand raising data to database: {str(db_error)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)} - Request ID: {request.state.request_id}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
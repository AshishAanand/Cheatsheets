# FastAPI Cheat Sheet

A comprehensive reference guide for building modern, fast web APIs with FastAPI.

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Basic App Structure](#basic-app-structure)
- [Path Operations](#path-operations)
- [Request & Response Models](#request--response-models)
- [Path Parameters](#path-parameters)
- [Query Parameters](#query-parameters)
- [Request Body](#request-body)
- [Response Models](#response-models)
- [Error Handling](#error-handling)
- [Dependencies](#dependencies)
- [Authentication & Security](#authentication--security)
- [Database Integration](#database-integration)
- [File Operations](#file-operations)
- [Background Tasks](#background-tasks)
- [WebSockets](#websockets)
- [Testing](#testing)
- [Deployment](#deployment)

## Installation & Setup

```bash
# Install FastAPI
pip install fastapi

# Install ASGI server (Uvicorn)
pip install uvicorn

# Install all optional dependencies
pip install fastapi[all]

# Run the application
uvicorn main:app --reload
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Basic App Structure

```python
from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI(
    title="My API",
    description="This is a sample API",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Basic route
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Sync function (also works)
@app.get("/sync")
def sync_root():
    return {"message": "Hello from sync"}

# Application events
@app.on_event("startup")
async def startup_event():
    print("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down...")
```

## Path Operations

```python
from fastapi import FastAPI

app = FastAPI()

# HTTP Methods
@app.get("/items")           # GET
async def read_items():
    return {"items": []}

@app.post("/items")          # POST
async def create_item():
    return {"message": "Item created"}

@app.put("/items/{item_id}") # PUT
async def update_item(item_id: int):
    return {"item_id": item_id}

@app.patch("/items/{item_id}") # PATCH
async def patch_item(item_id: int):
    return {"item_id": item_id}

@app.delete("/items/{item_id}") # DELETE
async def delete_item(item_id: int):
    return {"message": "Item deleted"}

# Multiple HTTP methods
@app.api_route("/items/{item_id}", methods=["GET", "POST"])
async def handle_item(item_id: int):
    return {"item_id": item_id}

# Route with tags and metadata
@app.get(
    "/items/{item_id}",
    tags=["items"],
    summary="Get an item",
    description="Get an item by its ID",
    response_description="The item details"
)
async def get_item(item_id: int):
    return {"item_id": item_id}
```

## Request & Response Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Request model
class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    price: float = Field(..., gt=0)
    tax: Optional[float] = Field(None, ge=0)
    tags: List[str] = []

# Response model
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    tax: Optional[float]
    created_at: datetime

    class Config:
        # Enable ORM mode for database models
        orm_mode = True

# Nested models
class User(BaseModel):
    id: int
    username: str
    email: str

class ItemWithOwner(Item):
    owner: User

# Model with example
class ItemExample(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "name": "Laptop",
                "description": "Gaming laptop",
                "price": 1200.00,
                "tax": 120.00
            }
        }
```

## Path Parameters

```python
# Basic path parameter
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

# Multiple path parameters
@app.get("/users/{user_id}/items/{item_id}")
async def get_user_item(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}

# Path parameter with validation
from fastapi import Path

@app.get("/items/{item_id}")
async def get_item(
    item_id: int = Path(..., title="Item ID", ge=1, le=1000)
):
    return {"item_id": item_id}

# Enum path parameters
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name}

# File path parameter
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
```

## Query Parameters

```python
from typing import Optional, List

# Basic query parameters
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 100):
    return {"skip": skip, "limit": limit}

# Optional query parameters
@app.get("/items/{item_id}")
async def get_item(item_id: int, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

# Query parameter validation
from fastapi import Query

@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(
        None,
        title="Query string",
        description="Query string for the items to search",
        min_length=3,
        max_length=50,
        regex="^[a-zA-Z0-9 ]+$"
    )
):
    return {"q": q}

# Multiple query parameters with same name
@app.get("/items/")
async def read_items(tags: List[str] = Query([])):
    return {"tags": tags}

# Required query parameter
@app.get("/items/")
async def read_items(required_param: str = Query(...)):
    return {"required_param": required_param}

# Boolean query parameters
@app.get("/items/")
async def read_items(active: bool = True):
    return {"active": active}
```

## Request Body

```python
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Single request body
@app.post("/items/")
async def create_item(item: Item):
    return item

# Request body + path parameters
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}

# Request body + path + query parameters
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    q: Optional[str] = None
):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

# Multiple request bodies
class User(BaseModel):
    username: str
    full_name: Optional[str] = None

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, user: User):
    return {"item_id": item_id, "item": item, "user": user}

# Body with additional validation
from fastapi import Body

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    importance: int = Body(..., gt=0, le=10)
):
    return {"item_id": item_id, "item": item, "importance": importance}
```

## Response Models

```python
from typing import List

# Response model
@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate):
    # Process item creation
    return Item(
        id=1,
        name=item.name,
        description=item.description,
        price=item.price,
        tax=item.tax,
        created_at=datetime.now()
    )

# Response model with list
@app.get("/items/", response_model=List[Item])
async def read_items():
    return [
        Item(id=1, name="Item 1", price=10.0, created_at=datetime.now()),
        Item(id=2, name="Item 2", price=20.0, created_at=datetime.now())
    ]

# Exclude fields from response
@app.get("/items/{item_id}", response_model=Item, response_model_exclude={"tax"})
async def get_item(item_id: int):
    return Item(id=item_id, name="Item", price=10.0, tax=1.0, created_at=datetime.now())

# Include only specific fields
@app.get("/items/{item_id}", response_model=Item, response_model_include={"name", "price"})
async def get_item_basic(item_id: int):
    return Item(id=item_id, name="Item", price=10.0, tax=1.0, created_at=datetime.now())

# Different response models for different status codes
from fastapi.responses import JSONResponse

@app.post("/items/", responses={
    201: {"model": Item, "description": "Item created"},
    400: {"description": "Bad request"},
    422: {"description": "Validation error"}
})
async def create_item(item: ItemCreate):
    if item.price < 0:
        return JSONResponse(
            status_code=400,
            content={"message": "Price cannot be negative"}
        )
    return Item(id=1, **item.dict(), created_at=datetime.now())
```

## Error Handling

```python
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Raise HTTP exceptions
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id == 0:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Item does not exist"}
        )
    return {"item_id": item_id}

# Custom exception
class ItemNotFoundException(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

@app.exception_handler(ItemNotFoundException)
async def item_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": f"Item {exc.item_id} not found"}
    )

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"message": "Validation error", "details": exc.errors()}
    )

# Using status codes from status module
@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    return {"message": "Item created"}
```

## Dependencies

```python
from fastapi import Depends

# Simple dependency
def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# Class-based dependency
class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    return commons

# Dependency with dependency
def get_db():
    db = DatabaseSession()
    try:
        yield db
    finally:
        db.close()

def get_current_user(db: Session = Depends(get_db)):
    # Get user from database
    return {"user_id": 1, "username": "john"}

@app.get("/users/me")
async def read_current_user(current_user: dict = Depends(get_current_user)):
    return current_user

# Application-level dependencies
app = FastAPI(dependencies=[Depends(common_parameters)])

# Router-level dependencies
from fastapi import APIRouter

router = APIRouter(dependencies=[Depends(get_current_user)])

# Dependency with yield (cleanup)
def get_database():
    db = Database()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def read_items(db: Database = Depends(get_database)):
    return db.get_items()
```

## Authentication & Security

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

# HTTP Bearer Token
security = HTTPBearer()

@app.get("/protected")
async def protected_route(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return {"token": credentials.credentials}

# OAuth2 Password Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/protected")
async def protected_route(token: str = Depends(oauth2_scheme)):
    return {"token": token}

# JWT Token authentication
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    # Get user from database
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

## Database Integration

```python
# SQLAlchemy setup
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class ItemDB(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Float)

Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CRUD operations
@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = ItemDB(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/", response_model=List[Item])
async def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = db.query(ItemDB).offset(skip).limit(limit).all()
    return items

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemDB).filter(ItemDB.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

# Database with Tortoise ORM (Async)
from tortoise.models import Model
from tortoise import fields
from tortoise.contrib.fastapi import register_tortoise

class ItemTortoise(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=100)
    description = fields.TextField(null=True)
    price = fields.FloatField()

register_tortoise(
    app,
    db_url="sqlite://db.sqlite3",
    modules={"models": ["__main__"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

@app.post("/items/")
async def create_item_tortoise(item: ItemCreate):
    item_obj = await ItemTortoise.create(**item.dict())
    return item_obj
```

## File Operations

```python
from fastapi import File, UploadFile, Form
from typing import List
import aiofiles

# Single file upload
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type}

# Multiple files upload
@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return [{"filename": file.filename} for file in files]

# File with additional form data
@app.post("/files/")
async def create_file(
    file: bytes = File(...),
    fileb: UploadFile = File(...),
    token: str = Form(...)
):
    return {
        "file_size": len(file),
        "token": token,
        "fileb_content_type": fileb.content_type,
    }

# Save uploaded file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    async with aiofiles.open(f"uploads/{file.filename}", 'wb') as f:
        content = await file.read()
        await f.write(content)
    return {"filename": file.filename, "size": len(content)}

# File download
from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"uploads/{filename}"
    return FileResponse(file_path, filename=filename)

# Stream file response
from fastapi.responses import StreamingResponse
import io

@app.get("/generate-csv")
async def generate_csv():
    def generate():
        data = io.StringIO()
        data.write("id,name,price\n")
        for i in range(1000):
            data.write(f"{i},Item {i},{i * 10}\n")
        data.seek(0)
        yield data.read()
    
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=items.csv"}
    )
```

## Background Tasks

```python
from fastapi import BackgroundTasks
import smtplib
from email.mime.text import MIMEText

def send_email(email: str, message: str):
    # Simulate sending email
    print(f"Sending email to {email}: {message}")

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Account created successfully")
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification sent in the background"}

# Background task with dependencies
def process_data(item_id: int, db: Session):
    # Process data
    pass

@app.post("/process/{item_id}")
async def start_processing(
    item_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    background_tasks.add_task(process_data, item_id, db)
    return {"message": "Processing started"}
```

## WebSockets

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")

# WebSocket with authentication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    # Validate token
    user = validate_token(token)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await websocket.accept()
    # Handle WebSocket communication
```

## Testing

```python
# test_main.py
from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 10.0}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"

def test_invalid_item():
    response = client.post(
        "/items/",
        json={"name": "", "price": -1}
    )
    assert response.status_code == 422

# Async testing with pytest-asyncio
@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200

# Testing with database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

def test_create_item_with_db():
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 10.0}
    )
    assert response.status_code == 200
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

```bash
# Build and run
docker build -t myapi .
docker run -d --name myapi-container -p 80:80 myapi
```

### Production Configuration

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My API"
    admin_email: str
    secret_key: str
    database_url: str
    
    class Config:
        env_file = ".env"

settings = Settings()

# main.py
from config import settings

app = FastAPI(title=settings.app_name)
```

```bash
# .env file
ADMIN_EMAIL=admin@example.com
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### Uvicorn Production

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# With configuration file
gunicorn -c gunicorn.conf.py main:app
```

```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
```

## Essential Tips & Best Practices

### Project Structure
```
app/
├── __init__.py
├── main.py
├── config.py
├── database.py
├── models/
│   ├── __init__.py
│   ├── user.py
│   └── item.py
├── routers/
│   ├── __init__.py
│   ├── users.py
│   └── items.py
├── dependencies.py
├── security.py
└── tests/
    ├── __init__.py
    └── test_main.py
```

### Performance Tips
```python
# Use async/await for I/O operations
@app.get("/items/")
async def read_items():
    # Use async database calls
    items = await db.fetch_all("SELECT * FROM items")
    return items

# Use dependencies for common logic
@app.get("/items/", dependencies=[Depends(rate_limit)])
async def read_items():
    return {"items": []}

# Use response models to limit data exposure
@app.get("/users/", response_model=List[UserPublic])
async def read_users():
    return users
```

## Useful Resources

- **Official Documentation:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Interactive API Docs:** Available at `/docs` when running your app
- **Alternative Docs:** Available at `/redoc` when running your app
- **GitHub:** [github.com/tiangolo/fastapi](https://github.com/tiangolo/fastapi)
- **Full Stack Template:** [github.com/tiangolo/full-stack-fastapi-postgresql](https://github.com/tiangolo/full-stack-fastapi-postgresql)
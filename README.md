# HRS
Health Recommendation System

A Flask-based web application that allows users to input text and displays the same text on an output screen. The application runs using Gunicorn as the WSGI server and can be deployed as a Docker container.

## Features

- Clean and responsive web interface
- Text input form with validation
- Output screen displaying the submitted text
- Production-ready with Gunicorn WSGI server
- Dockerized for easy deployment

## Prerequisites

- Python 3.13 or higher
- Docker (for containerized deployment)

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run with Flask development server:
```bash
python app.py
```

3. Run with Gunicorn (Linux only):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

4. Open your browser and navigate to `http://localhost:5000`

## Running with Docker

1. Build the Docker image:
```bash
docker build -t hrs-flask-app .
```

2. Run the container:
```bash
docker run -d -p 8080:5000 --name hrs hrs-flask-app
```

3. Open your browser and navigate to `http://localhost:8080`

4. Stop the container:
```bash
docker stop hrs
docker rm hrs
```

## Project Structure

```
HRS/
├── app.py                  # Flask application
├── templates/
│   ├── index.html         # Input form page
│   └── output.html        # Output display page
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── .dockerignore        # Docker ignore file
└── README.md            # This file
```

## Usage

1. Enter text in the input box on the home page
2. Click "Submit"
3. View your text displayed on the output page
4. Click "Go Back" to enter new text

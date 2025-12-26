# Hand-Written Digit Recognizer

An interactive web application that uses a Convolutional Neural Network (CNN) to recognize hand-written digits (0-9). Users can draw digits on a canvas, and the AI model will predict which digit was drawn with confidence scores.

## Features

- 🎨 **Interactive Drawing Canvas**: Draw digits naturally with your mouse or touchscreen
- 🤖 **Deep Learning**: CNN model trained on the MNIST dataset
- 📊 **Confidence Visualization**: See probability distributions for all digits
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 🌐 **Web-Based**: Accessible from any device with a web browser
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
.
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore          # Docker ignore rules
├── templates/
│   └── index.html         # Main web interface
├── static/
│   ├── style.css          # Styling
│   └── app.js             # Canvas and interaction logic
└── mnist_model.h5         # Trained model (generated)
```

## Prerequisites

Choose one of the following:

### Option 1: Local Development
- Python 3.12
- pip or uv package manager

### Option 2: Docker Deployment
- Docker
- Docker Compose

## Installation & Setup

### Option 1: Local Development

1. **Clone or navigate to the project directory**

2. **Install dependencies**

   Using uv (recommended):
   ```bash
   uv add -r requirements.txt
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: You'll also need Graphviz installed on your system for model visualization:
   ```bash
   # macOS
   brew install graphviz

   # Ubuntu/Debian
   sudo apt-get install graphviz

   # Windows (with Chocolatey)
   choco install graphviz
   ```

3. **Train the model**

   ```bash
   uv run python train_model.py
   ```

   This will:
   - Download the MNIST dataset
   - Train a CNN model (10 epochs, ~5 minutes)
   - Save the model as `mnist_model.h5`
   - Achieve ~99% test accuracy

4. **Run the application**

   ```bash
   uv run python app.py
   ```

   Or with Flask directly:
   ```bash
   flask run --host=0.0.0.0 --port=5001
   ```

5. **Access the application**

   Open your browser to: http://localhost:5001

### Option 2: Docker Deployment

1. **Train the model first** (if not already trained)

   ```bash
   uv run python train_model.py
   ```

   This ensures `mnist_model.h5` exists before building the Docker image.

2. **Build and run with Docker Compose**

   ```bash
   docker-compose up -d
   ```

   This will:
   - Build the Docker image
   - Start the container
   - Expose the app on port 5001

3. **Access the application**

   Open your browser to: http://localhost:5001

4. **View logs**

   ```bash
   docker-compose logs -f
   ```

5. **Stop the application**

   ```bash
   docker-compose down
   ```

### Manual Docker Build

If you prefer to use Docker without Docker Compose:

```bash
# Build the image
docker build -t digit-recognizer .

# Run the container
docker run -d -p 5001:5001 \
  -v $(pwd)/mnist_model.h5:/app/mnist_model.h5:ro \
  --name digit-recognizer \
  digit-recognizer

# Stop the container
docker stop digit-recognizer
docker rm digit-recognizer
```

## Usage

1. **Draw a digit** (0-9) on the white canvas using your mouse or finger
2. **Click "Recognize Digit"** to get the prediction
3. **View results**:
   - The predicted digit (large display)
   - Confidence percentage
   - Probability distribution for all digits (0-9)
4. **Click "Clear"** to reset and draw another digit

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Input**: 28×28 grayscale images
- **Layers**:
  - 3 Convolutional layers with MaxPooling
  - Dropout for regularization
  - Fully connected dense layers
  - Softmax output (10 classes)
- **Accuracy**: ~99% on test set

## API Endpoints

### `GET /`
Serves the main web interface.

### `POST /predict`
Accepts JSON with base64-encoded canvas image and returns prediction.

**Request:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "digit": 7,
  "confidence": 0.9856,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    ...
    "7": 0.9856,
    ...
  }
}
```

### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Deployment to Production

### Web Server Configuration

For production deployment on your private website:

1. **Use a reverse proxy** (nginx/Apache) to forward requests to the Docker container
2. **Configure SSL/TLS** for HTTPS
3. **Set up domain routing** to your application

Example nginx configuration:

```nginx
server {
    listen 80;
    server_name digits.yourdomain.com;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Docker Production Tips

- Use `docker-compose` with `restart: unless-stopped` (already configured)
- Monitor container health with the built-in health check
- Consider using Docker volumes for persistent model storage
- Set appropriate resource limits in production

## Troubleshooting

### Model not found error
- Run `uv run python train_model.py` to train and save the model
- Ensure `mnist_model.h5` exists in the project directory

### Canvas not responding
- Check browser console for JavaScript errors
- Ensure static files are being served correctly

### Low prediction accuracy
- Make sure you draw digits clearly
- Try drawing thicker lines (similar to MNIST training data)
- Center your digit in the canvas

### Docker issues
- Ensure the model file exists before building the image
- Check container logs: `docker-compose logs -f`
- Verify port 5001 is not already in use

## License

MIT License - feel free to use this project for learning and deployment.

## Acknowledgments

- MNIST dataset
- TensorFlow and Keras teams
- Flask web framework

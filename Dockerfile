# Slim demo image — boots on the lightweight deps (no torch/chromadb).
# Build:  docker build -t rag-optimized .
# Run:    docker run -p 8000:8000 rag-optimized   ->  http://localhost:8000
#
# For full real-RAG mode, swap requirements.txt for requirements-full.txt below
# and pass your key:  docker run -e ANTHROPIC_API_KEY=sk-... -p 8000:8000 rag-optimized
FROM python:3.12-slim

WORKDIR /app

# Install deps first so Docker caches this layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app.
COPY . .

# Render/Fly/Railway inject $PORT; default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]

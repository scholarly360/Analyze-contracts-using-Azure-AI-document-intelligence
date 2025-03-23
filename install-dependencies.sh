sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && sudo rm -rf /var/lib/apt/lists/*
sudo apt install poppler-utils -y
python -m pip install --no-cache-dir Flask python-dotenv pdf2image azure-ai-documentintelligence openai
##pdftotext -v

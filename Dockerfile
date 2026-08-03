FROM dailyco/pipecat-base:latest

ENV PYTHONPATH=/app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
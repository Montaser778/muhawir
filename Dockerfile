FROM dailyco/pipecat-base:latest

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN ls -la /app && echo "---" && ls -la /app/interview /app/static
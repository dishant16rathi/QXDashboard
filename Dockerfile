FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r req.txt
RUN pip install --no-cache-dir numpy==1.24.3
# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["./startup.sh"]
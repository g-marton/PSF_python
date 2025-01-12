FROM python:3.11-slim

# Install Microsoft Fonts and other dependencies
RUN apt-get update && apt-get install -y \
    ttf-mscorefonts-installer \
    fontconfig && \
    fc-cache -f -v && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "PSF_analysis_appvol1.py", "--server.headless", "true"]

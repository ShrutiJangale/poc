# POC Projects Overview

This folder contains multiple  projects. This document provides a comprehensive guide to setting up and running each project.

##  Projects List

1. **deltaAnalysis** - Procurement and True-Up Data Analysis using OpenAI
2. **demo-supplychain-ai** - Supply Chain AI Analysis Tool
3. **royalFinanceDemo** - Bank Statement Analyzer with AI
4. **VehicleDetection** - Vehicle Detection and Counting using YOLO

---

##  Common Setup Instructions

### Prerequisites

- Python 
- pip (Python package installer)
- Virtual environment support

### Setting Up Virtual Environment (Common for All Projects)

Each project will share a common venv

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies:**
   pip install -r requirements.txt

## 📋 Project Details

### 1. deltaAnalysis

**Description:** A  application for analyzing procurement and true-up data using semantic matching with OpenAI. It matches items between procurement sheets and true-up sheets using AI-powered semantic similarity.

**Location:** `poc/deltaAnalysis/`

**Features:**
- Excel file upload and processing
- Semantic matching between procurement and true-up data
- Delta analysis and reporting

**Setup Steps:**

1. **Navigate to project directory:**
   ```bash
   cd poc/deltaAnalysis
   ```
2. **Create `.env` file:**

   Create a `.env` file in the `deltaAnalysis` directory with the following content:
   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

4. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

5. **Run development server:**
   ```bash
   python manage.py runserver
   ```

6. **Access the application:**
   Open your browser and navigate to: http://127.0.0.1:8000/


### 2. demo-supplychain-ai

**Description:** A application for analyzing supply chain data using AI-powered insights. It processes supply chain data files and provides AI-generated summaries and analysis.

**Location:** `poc/demo-supplychain-ai/supplychain_ai/`

**Features:**
- Supply chain data file upload
- AI-powered analysis and summarization
- Dashboard for viewing processed data

**Setup Steps:**

1. **Navigate to project directory:**
   ```bash
   cd poc/demo-supplychain-ai/supplychain_ai
   ```

2. **Create `.env` file:**
   Create a `.env` file in the `supplychain_ai` directory (same level as `manage.py`) with the following content:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   DJANGO_SECRET_KEY=your-secret-key-here (optional, has default)
   DJANGO_DEBUG=true (optional, defaults to true)
   DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 (optional)
   ```

3. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

4. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

5. **Run development server:**
   ```bash
   python manage.py runserver
   ```

6. **Access the application:**
   Open your browser and navigate to: http://127.0.0.1:8000/


### 3. royalFinanceDemo

**Description:** A application for analyzing bank statements using AI. It extracts data from PDF bank statements, processes transactions, and provides AI-powered insights and verification.

**Location:** `poc/royalFinanceDemo/`

**Features:**
-  bank statement upload and parsing
- Transaction extraction and analysis
- AI-powered transaction verification
- Data enhancement and categorization

**Setup Steps:**

1. **Navigate to project directory:**
   ```bash
   cd poc/royalFinanceDemo
   ```

2. **Create `.env` file:**
   Create a `.env` file in the `royalFinanceDemo` directory (same level as `manage.py`) with the following content:
   ```env
   KEY_OPENAI=your-openai-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here
   OPEN_AI_MODEL=gpt-4 or gpt-3.5-turbo
   MAX_TOKEN_LIMIT=4096
   ```

3. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

4. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

5. **Run development server:**
   ```bash
   python manage.py runserver
   ```

6. **Access the application:**
   Open your browser and navigate to: http://127.0.0.1:8000/


### 4. VehicleDetection

**Description:** A application for vehicle detection and counting in videos using YOLO object detection and DeepSort tracking. It can detect and count vehicles (cars, trucks, buses) in video files.

**Location:** `poc/VehicleDetection/`

**Features:**
- Video upload and processing
- Real-time vehicle detection using YOLO
- Vehicle tracking using DeepSort
- Vehicle counting across a defined line

**Setup Steps:**

1. **Navigate to project directory:**
   ```bash
   cd poc/VehicleDetection
   ```

2. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

3. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

4. **Run development server:**
   ```bash
   python manage.py runserver
   ```

5. **Access the application:**
   Open your browser and navigate to: http://127.0.0.1:8000/





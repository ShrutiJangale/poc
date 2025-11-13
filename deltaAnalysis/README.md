# Delta Analysis Project

A Django web application for analyzing procurement and true-up data using semantic matching with OpenAI.

## Setup Instructions

### 1. Navigate to Project Directory

First, navigate to the `deltaAnalysis` folder:

```bash
cd poc/deltaAnalysis
```

**Windows (PowerShell):**
```powershell
cd poc\deltaAnalysis
```

**Windows (Command Prompt):**
```cmd
cd poc\deltaAnalysis
```

### 2. Set Up Virtual Environment

If you haven't already created a virtual environment, create one:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure you're in the `deltaAnalysis` directory and your virtual environment is activated:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

1. Create a `.env` file in the `deltaAnalysis` directory (if it doesn't exist)

2. Edit `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

   **Get your API key from:** https://platform.openai.com/api-keys

### 5. Run Database Migrations

```bash
python manage.py migrate
```

### 6. Create a Superuser (Optional)

If you want to access the Django admin panel:

```bash
python manage.py createsuperuser
```

### 7. Run the Development Server

```bash
python manage.py runserver
```

The application will be available at: http://127.0.0.1:8000/

## Project Structure

- `analysis_app/` - Main Django app for file upload and analysis
- `psananalysis_project/` - Django project settings
- `Files/` - Sample Excel files for testing
- `manage.py` - Django management script

## Usage

1. Navigate to http://127.0.0.1:8000/
2. Upload two Excel files:
   - Procurement Sheet (should contain: Description, Unit Cost, Quantity Ordered columns)
   - True Up Sheet (should contain: Type, Description, QTY, Size columns)
3. The application will analyze and match items using semantic similarity
4. View the delta analysis results

## Notes

- Make sure your Excel files have the correct column names
- The application uses OpenAI API for semantic matching (costs apply)
- Debug mode is enabled by default (DEBUG=True in settings.py)


# Delta Analysis Project

A Django web application for analyzing procurement and true-up data using semantic matching with OpenAI.

## Setup Instructions

### 1. Activate Virtual Environment

If you haven't already activated your virtual environment:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

### 2. Install Dependencies

Make sure you're in the `DeltaAnalysis` directory and your virtual environment is activated:

```bash
cd DeltaAnalysis
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

   **Get your API key from:** https://platform.openai.com/api-keys

### 4. Run Database Migrations

```bash
python manage.py migrate
```

### 5. Create a Superuser (Optional)

If you want to access the Django admin panel:

```bash
python manage.py createsuperuser
```

### 6. Run the Development Server

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


## RoyalFinanceAI -  Bank Statement Analyzer

Step-by-step guide to set up and run the app locally.

### 1) Create and activate a virtual environment 
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
cd .\royalfinanceAI\
pip install -r requirements.txt
```

### 2) Navigate to project folder
```bash
cd .\royalfinanceAI\
```

### 4) Environment variables
Create a `.env` file next to `manage.py` at `royalfinanceAI\.env` if needed. Example:
```
KEY_OPENAI = ""
GEMINI_API_KEY=""
OPEN_AI_MODEL = ""
MAX_TOKEN_LIMIT = 

```

### 5) Run database migrations
```bash
python manage.py migrate
```

 create a superuser for the admin panel
```bash
python manage.py createsuperuser
```

### 6) Start the development server
```bash
python manage.py runserver
```
Open your browser at:  http://127.0.0.1:8000/




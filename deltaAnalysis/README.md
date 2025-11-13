# Delta Analysis Project

A Django web application for analyzing procurement and true-up data using semantic matching with OpenAI

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
```bash
pip install -r requirements.txt
```

### 2. Navigate to project folder
```bash
cd DeltaAnalysis
```

### 3. Set Up Environment Variables

 Edit `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

### 3. Run Database Migrations

```bash
python manage.py migrate
```

### 4. Create a Superuser (Optional)

If you want to access the Django admin panel:

```bash
python manage.py createsuperuser
```

### 5. Run the Development Server

```bash
python manage.py runserver
```

The application will be available at: http://127.0.0.1:8000/



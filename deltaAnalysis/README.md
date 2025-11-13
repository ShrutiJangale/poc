# Delta Analysis Project

A Django web application for analyzing procurement and true-up data using semantic matching with OpenAI

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
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure you're in the `deltaAnalysis` directory and your virtual environment is activated:

```bash
pip install -r requirements.txt
=======
### 2. Navigate to project folder
```bash
cd DeltaAnalysis
>>>>>>> f0211e100caee46c3a057187c4c42ecb1c2549e1
```

### 4. Set Up Environment Variables

<<<<<<< HEAD
1. Create a `.env` file in the `deltaAnalysis` directory (if it doesn't exist)

2. Edit `.env` file and add your OpenAI API key:
=======
 Edit `.env` file and add your OpenAI API key:
>>>>>>> f0211e100caee46c3a057187c4c42ecb1c2549e1
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

<<<<<<< HEAD
   **Get your API key from:** https://platform.openai.com/api-keys

### 5. Run Database Migrations
=======
### 3. Run Database Migrations
>>>>>>> f0211e100caee46c3a057187c4c42ecb1c2549e1

```bash
python manage.py migrate
```

<<<<<<< HEAD
### 6. Create a Superuser (Optional)
=======
### 4. Create a Superuser (Optional)
>>>>>>> f0211e100caee46c3a057187c4c42ecb1c2549e1

If you want to access the Django admin panel:

```bash
python manage.py createsuperuser
```

<<<<<<< HEAD
### 7. Run the Development Server
=======
### 5. Run the Development Server
>>>>>>> f0211e100caee46c3a057187c4c42ecb1c2549e1

```bash
python manage.py runserver
```

The application will be available at: http://127.0.0.1:8000/



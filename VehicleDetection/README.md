# Vehicle Detection Project

A Django web application for real-time vehicle detection and counting in video streams using YOLOv8 and DeepSort tracking

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

### 2. Navigate to project folder
```bash
cd VehicleDetection
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


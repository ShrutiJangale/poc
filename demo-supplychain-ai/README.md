# Supply Chain AI Analysis Tool

A Django-based web application for analyzing supply chain data using AI-powered insights.


# 1. Clone and navigate
git clone "https://github.com/ShrutiJangale/demo-supplychain-ai.git"
cd demo-supplychain-ai

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Navigate to project directory
cd supplychain_ai

# 5. Create .env file and add your OpenAI API key
# Create .env file in this directory and add: OPENAI_API_KEY=your-key-here

# 6. Run migrations
python manage.py migrate

# 7. Start server
python manage.py runserver
```

Visit http://127.0.0.1:8000/ to access the application.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))



### 5. Create Environment Variables File


demo-supplychain-ai/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── supplychain_ai/          # Main project directory
    ├── manage.py            # Django management script
    ├── .env                 # Environment variables (create this)
    ├── supplychain_ai/      # Django project settings
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── analysis_supplychain_ai/  # Main application
        ├── models.py
        ├── views.py
        ├── urls.py
        ├── templates/       # HTML templates
        ├── static/          # CSS and static files
        └── utils/           # Utility functions
```

## Usage

1. **Access the Dashboard**: Navigate to http://127.0.0.1:8000/
2. **Upload Files**: Use the upload interface to add supply chain data files
3. **View Results**: Analyze the processed data and view AI-generated summaries



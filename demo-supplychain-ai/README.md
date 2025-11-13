# Supply Chain AI Analysis Tool

A Django-based web application for analyzing supply chain data using AI-powered insights.

# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 4. Navigate to project directory
cd demo-supplychain-ai/supplychain_ai

# 5. Create .env file and add your OpenAI API key
# Create .env file in this directory and add: OPENAI_API_KEY=your-key-here

# 6. Run migrations
python manage.py migrate

# 7. Start server
python manage.py runserver
```


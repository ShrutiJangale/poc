from django.apps import AppConfig


class AnalysisSupplychainAiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analysis_supplychain_ai'
    
    def ready(self):
        """Initialize database schema when app is ready."""
        import sys
        # Prevent running during migrations and avoid duplicate initialization
        if len(sys.argv) > 1:
            command = sys.argv[1]
            if command in ['migrate', 'makemigrations', 'test', 'collectstatic']:
                return
        
        # Initialize database schema
        try:
            from .utils.db_utils import create_schema
            create_schema()
        except Exception as e:
            # Log error but don't crash the app
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize database schema: {e}")
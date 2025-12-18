#!/bin/bash

# Activate virtual environment if used
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Export Flask environment variables
export FLASK_APP=app/routes.py
export FLASK_ENV=development

# Run Flask application
flask run --host=0.0.0.0 --port=5000

# Deactivate virtual environment if used
if [ -d "venv" ]; then
    deactivate
fi

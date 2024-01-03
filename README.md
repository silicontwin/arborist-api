# Project Status
The **Arborist API** is currently in alpha development. We're actively working on adding core API functionality. The beta testing phase has not yet commenced, and the `Issues` tab for this repository will remain disabled until the app reaches the appropriate level of usability/polish.

---

# Initial Setup
- Create, and navigate to, a new directory for this project
- `python3 -m venv env`: Create a Python virtual environment named `env`
- Set the new virtual environment as the Python interpreter in your IDE
- `source env/bin/activate`: Activate the virtual environment
- `pip install -r requirements.txt`: Install the Python dependencies listed in `requirements.txt`
- `pip list`: List the installed Python packages
- `deactivate`: Deactivate the virtual environment
- `source env/bin/activate`: Reactivate the virtual environment (the freshly installed packages should now be available)
- `uvicorn app.main:app --reload`: Run the FastAPI server

# Running the FastAPI Server
- `source env/bin/activate`: Activate the virtual environment
- `uvicorn app.main:app --reload`: Run the FastAPI server

# API Documentation
- Follow the steps in the `Running the FastAPI Server` section above
- `http://localhost:8000/docs`: Open the Swagger UI to view endpoint documentation
- `http://localhost:8000/redoc`: Open the ReDoc UI to view endpoint documentation

---

# Testing (run these commands from within your venv)

- `python3 helpers/generate_test_spss.py`: Generate a test .spss file with 10K observations and 10 features
- `python3 helpers/generate_test_csv.py`: Generate a test .csv file with 10K observations and 10 features

---

# Preparing for AWS Lambda
When deploying to AWS Lambda, we'll use `mangum` to wrap our FastAPI application. We'll need to modify the `main.py` to include:

```python
from mangum import Mangum
# FastAPI app code here
handler = Mangum(app)
```

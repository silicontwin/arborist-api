# Status
- **Arborist** is currently in alpha development. We're actively working on adding core API functionality. The beta testing phase has not yet commenced, and the `Issues` tab for this repository will remain disabled until the app reaches the appropriate level of usability/polish.

---

# Instructions

- `python3 -m venv env` - Create a Python virtual environment named `env`
- Set the new virtual environment as the Python interpreter in your IDE
- `source env/bin/activate` - Activate the virtual environment
- `pip install -r requirements.txt` - Install the Python dependencies listed in `requirements.txt`
- `pip list` - List the installed Python packages
- `deactivate` - Deactivate the virtual environment
- `source env/bin/activate` - Reactivate the virtual environment (the freshly installed packages should now be available)
- `uvicorn app.main:app --reload` - Run the FastAPI server

---

# Testing (run these commands from within your venv)

- `python3 generate_test_spss.py` - Generate a test SPSS file to upload at `/`
- `python3 generate_test_csv.py` - Generate a test CSV file to upload at `/`

---

# Preparing for AWS Lambda
When deploying to AWS Lambda, we'll use `mangum` to wrap our FastAPI application. We'll need to modify the `main.py` to include:

```python
from mangum import Mangum
# FastAPI app code here
handler = Mangum(app)
```
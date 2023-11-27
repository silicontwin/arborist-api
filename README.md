
# Instructions

- `python3 -m venv env` - Create a Python virtual environment named `env`
- Set the new virtual environment as the Python interpreter in your IDE
- `source env/bin/activate` - Activate the virtual environment
- `pip install -r requirements.txt` - Install the Python dependencies listed in `requirements.txt`
- `uvicorn main:app --reload` - Run the FastAPI server

---

# Testing

- `python3 generate_test_spss.py` - Generate a test SPSS file to upload at `/`

---

# Preparing for AWS Lambda:
When deploying to AWS Lambda, we'll use `mangum` to wrap our FastAPI application. We'll need to modify the `main.py` to include:

```python
from mangum import Mangum

# FastAPI app code here

handler = Mangum(app)
```
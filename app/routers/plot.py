# app/routers/plot.py
from fastapi import APIRouter, Response
import pandas as pd
from plotnine import ggplot, aes, geom_boxplot
import tempfile
import os

router = APIRouter()

@router.get("/plot", summary="Generate and return a plot image", response_class=Response)

async def generate_plot():
    # Example DataFrame
    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'],
        'value': [1, 2, 3, 2, 5, 3, 4, 5, 6]
    })

    # Create a box plot
    plot = ggplot(df, aes(x='category', y='value')) + geom_boxplot()

    # Save the plot to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        plot.save(tmpfile.name, format='png', width=6, height=4, dpi=150)
        tmpfile.seek(0)  # Go to the beginning of the file
        content = tmpfile.read() # Read the image and send it as a response
        os.unlink(tmpfile.name) # Clean up the temp file
        return Response(content=content, media_type="image/png")

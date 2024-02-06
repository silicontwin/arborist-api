# app/routers/plot.py
import matplotlib
matplotlib.use('Agg')

from fastapi import APIRouter, Response, HTTPException, Query
from fastapi.responses import FileResponse
import pandas as pd
from plotnine import ggplot, aes, geom_boxplot
import tempfile
import os

router = APIRouter()

@router.get("/plot", summary="Generate and return a plot image", response_class=Response)
async def generate_plot(
    width: int = Query(6, description="Width of the plot"),
    height: int = Query(4, description="Height of the plot"),
    dpi: int = Query(150, description="DPI of the plot")
):
    try:
        # Example DataFrame
        df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'],
            'value': [1, 2, 3, 2, 5, 3, 4, 5, 6]
        })

        # Create a box plot
        plot = ggplot(df, aes(x='category', y='value')) + geom_boxplot()

        # Use a temp file for the plot
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='static/plots') as tmpfile:
            plot_path = tmpfile.name
            plot.save(plot_path, format='png', width=width, height=height, dpi=dpi)

        # Return the plot file
        return FileResponse(plot_path, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

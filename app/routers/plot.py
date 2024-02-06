# app/routers/plot.py
import matplotlib
matplotlib.use('Agg')

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse
import pandas as pd
from plotnine import ggplot, aes, geom_boxplot
import os

router = APIRouter()

# Ensure the plots directory exists
plots_dir = "static/plots"
os.makedirs(plots_dir, exist_ok=True)

@router.get("/plot", summary="Generate and return a plot image", response_class=Response)
async def generate_plot():
    # Example DataFrame
    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'],
        'value': [1, 2, 3, 2, 5, 3, 4, 5, 6]
    })

    # Create a box plot
    plot = ggplot(df, aes(x='category', y='value')) + geom_boxplot()

    # Define the filename and path to save the plot
    plot_filename = "boxplot.png"
    plot_path = os.path.join(plots_dir, plot_filename)

    # Save the plot
    plot.save(plot_path, format='png', width=6, height=4, dpi=150)

    # Return the plot file
    return FileResponse(plot_path)

# app/routers/summarize.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pyarrow.dataset as ds
import os
import pandas as pd
import numpy as np
from stochtree import BARTModel
import logging
from typing import List, Optional

# Custom Formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        # custom_message = f"• {record.msg} | {record.levelname}:{record.name}"
        # custom_message = f"• {record.msg} | {record.name}"
        custom_message = f"• {record.msg}"
        return custom_message

# Configure logging
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = CustomFormatter()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

router = APIRouter()

class FileProcessRequest(BaseModel):
    fileName: str  # Can also be a directory of CSVs
    workspacePath: str
    selectedColumns: List[str] = []
    outcomeVariable: Optional[str] = None
    headTailRows: int = 20
    action: str = "summarize"

@router.post("/summarize")
async def read_data(request: FileProcessRequest):
    try:
        num_rows_to_display = request.headTailRows
        logger.debug(f"Number of rows to display: {num_rows_to_display}")

        # Construct the full file path using the workspacePath and fileName
        file_path = os.path.join(request.workspacePath, request.fileName)
        logger.debug(f"File path: {file_path}")

        # Check if the file or directory exists
        if not os.path.exists(file_path):
            error_msg = "File or directory not found"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        try:
            # Load the dataset from the file or directory of CSV files
            dataset = ds.dataset(file_path, format='csv')
            logger.debug("Dataset loaded successfully")
        except Exception as e:
            error_msg = f"Failed to read the dataset: {e}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Convert the dataset to a PyArrow table
        table = dataset.to_table()
        logger.debug("Dataset converted to PyArrow table")

        # Convert the table to a Pandas DataFrame for easier JSON serialization
        df = table.to_pandas()
        logger.debug("PyArrow table converted to Pandas DataFrame")

        # Select only the requested columns
        if request.selectedColumns:
            missing_columns = set(request.selectedColumns) - set(df.columns)
            if missing_columns:
                error_msg = f"Selected columns not found in the data: {missing_columns}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            df = df[request.selectedColumns]
            logger.debug(f"Selected columns: {request.selectedColumns}")

        # Store the initial number of rows before removing NaN values
        initial_row_count = len(df)
        logger.debug(f"Initial row count: {initial_row_count}")

        # Handle missing values: remove rows with NaN values
        df_cleaned = df.dropna()
        logger.debug("Rows with NaN values removed")

        # Calculate the number of observations removed
        observations_removed = initial_row_count - len(df_cleaned)
        logger.debug(f"Number of observations removed: {observations_removed}")

        response_data = {}

        if request.action == "analyze":
            logger.debug("Action: analyze")
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            logger.debug(f"Numeric columns: {numeric_cols}")
            if numeric_cols:
                if request.outcomeVariable not in df_cleaned.columns:
                    error_msg = f"Selected outcome variable '{request.outcomeVariable}' not found in the data"
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)

                # Select the user-specified features and outcome variable
                features = [col for col in request.selectedColumns if col != request.outcomeVariable]
                logger.debug(f"Features: {features}")
                outcome = request.outcomeVariable
                logger.debug(f"Outcome variable: {outcome}")

                X = df_cleaned[features].to_numpy()
                y = df_cleaned[outcome].to_numpy()
                logger.debug(f"X shape: {X.shape}")
                logger.debug(f"y shape: {y.shape}")
                logger.debug(f"X values: {X[:5]}")
                logger.debug(f"y values: {y[:5]}")

                # Additional logging to check for variability in X and y
                logger.debug(f"X mean: {np.mean(X, axis=0)}")
                logger.debug(f"X std: {np.std(X, axis=0)}")
                logger.debug(f"y mean: {np.mean(y)}")
                logger.debug(f"y std: {np.std(y)}")

                # Create an instance of BARTModel
                model = BARTModel()

                # Log BART model parameters
                logger.debug(f"BART model parameters: num_trees=100, num_gfr=10, num_mcmc=100")

                # Create a dummy basis array with a single column filled with zeros
                basis_train = np.zeros((X.shape[0], 1))

                # Sample the BART model
                model.sample(X_train=X, y_train=y, basis_train=basis_train, num_trees=100, num_gfr=10, num_mcmc=100)
                logger.debug("Model training completed")

                # Create a dummy basis array for prediction
                basis_pred = np.zeros((X.shape[0], 1))

                # Predict using the trained model
                y_pred = model.predict(covariates=X, basis=basis_pred)
                logger.debug(f"y_pred shape: {y_pred.shape}")
                logger.debug(f"y_pred values: {y_pred[:5]}")

                # Check for variability in y_pred
                logger.debug(f"y_pred mean: {np.mean(y_pred)}")
                logger.debug(f"y_pred std: {np.std(y_pred)}")

                if np.all(y_pred == y_pred[0]):
                    logger.error("Predictions are all the same. Model may not be training correctly.")
                    raise HTTPException(status_code=500, detail="Model predictions are not varying")

                # Transpose the y_pred array
                y_pred_transposed = y_pred.T
                logger.debug(f"y_pred_transposed shape: {y_pred_transposed.shape}")
                logger.debug(f"y_pred_transposed values: {y_pred_transposed[:, :5]}")

                # Ensure the length of y_pred_transposed matches the DataFrame's index length
                if y_pred_transposed.shape[1] != len(df_cleaned):
                    error_msg = f"Length of y_pred_transposed ({y_pred_transposed.shape[1]}) does not match length of DataFrame ({len(df_cleaned)})"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)

                # Compute posterior summaries
                posterior_mean = y_pred_transposed.mean(axis=0)
                percentile_2_5 = np.percentile(y_pred_transposed, 2.5, axis=0)
                percentile_97_5 = np.percentile(y_pred_transposed, 97.5, axis=0)

                logger.debug(f"posterior_mean: {posterior_mean[:5]}")
                logger.debug(f"percentile_2_5: {percentile_2_5[:5]}")
                logger.debug(f"percentile_97_5: {percentile_97_5[:5]}")

                # Prepend the posterior summary columns to the DataFrame
                df_cleaned.insert(0, '97.5th percentile', percentile_97_5)
                df_cleaned.insert(0, '2.5th percentile', percentile_2_5)
                df_cleaned.insert(0, 'Posterior Average (y hat)', posterior_mean)

                logger.debug("Posterior summaries added to DataFrame")
            else:
                error_msg = "No numeric columns found for analysis"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

        # Adjust DataFrame to include only a subset of rows based on num_rows_to_display
        if len(df_cleaned) > 2 * num_rows_to_display:
            placeholder = pd.DataFrame({col: ['...'] for col in df_cleaned.columns}, index=[0])
            df_final = pd.concat([df_cleaned.head(num_rows_to_display), placeholder, df_cleaned.tail(num_rows_to_display)], ignore_index=True)
        else:
            df_final = df_cleaned
        logger.debug("DataFrame adjusted based on num_rows_to_display")

        # Convert the DataFrame to JSON
        json_data = df_final.to_dict(orient='records')
        response_data["data"] = json_data
        response_data["wrangle"] = {"observationsRemoved": observations_removed}
        response_data["selectedColumns"] = request.selectedColumns
        logger.debug("Response data prepared")

        # Determine if each column is numeric and store the result in a dictionary
        is_numeric = {col: pd.api.types.is_numeric_dtype(df_cleaned[col]) for col in df_cleaned.columns}
        response_data["is_numeric"] = is_numeric
        logger.debug("Column numeric status determined")

        logger.debug("Request processed successfully")
        return response_data
    except Exception as e:
        error_msg = f"General error in processing file or directory: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

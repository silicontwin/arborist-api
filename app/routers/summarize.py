# app/routers/summarize.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pyarrow.dataset as ds
import os
import pandas as pd
import numpy as np
from app.model import BartModel
import logging
from typing import List, Optional
from stochtree import Dataset, Residual, RNG, ForestSampler, ForestContainer, GlobalVarianceModel, LeafVarianceModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

class FileProcessRequest(BaseModel):
    fileName: str  # Can also be a directory of CSVs
    workspacePath: str
    selectedColumns: List[str] = []  # Columns to be processed
    outcomeVariable: Optional[str] = None  # Outcome variable
    headTailRows: int = 20  # Number of head and tail observations to display
    action: str = "summarize"  # Default action is summarize

# Instantiate the model
model = BartModel()

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
            # Select only numeric columns for the BART model
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            logger.debug(f"Numeric columns: {numeric_cols}")
            if numeric_cols:
                if request.outcomeVariable not in df_cleaned.columns:
                    error_msg = f"Selected outcome variable '{request.outcomeVariable}' not found in the data"
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)

                X = df_cleaned[numeric_cols].to_numpy()
                X = np.ascontiguousarray(X)
                y = df_cleaned[request.outcomeVariable].to_numpy()  # Use the selected outcome variable
                logger.debug(f"X shape: {X.shape}")
                logger.debug(f"y shape: {y.shape}")

                # Standardize outcome
                y_bar = np.mean(y)
                y_std = np.std(y)
                resid = (y - y_bar) / y_std
                logger.debug("Outcome standardized")

                # Convert data to StochTree representation
                dataset = Dataset()
                dataset.add_covariates(X)
                residual = Residual(resid)
                logger.debug("Data converted to StochTree representation")

                # Set sampling parameters
                alpha = 0.9
                beta = 1.25
                min_samples_leaf = 1
                num_trees = 100
                cutpoint_grid_size = 100
                global_variance_init = 1.0
                tau_init = 0.5
                leaf_prior_scale = np.array([[tau_init]], order='C')
                nu = 4.0
                lamb = 0.5
                a_leaf = 2.0
                b_leaf = 0.5
                leaf_regression = True
                feature_types = np.repeat(0, X.shape[1]).astype(int)  # 0 = numeric
                var_weights = np.repeat(1 / X.shape[1], X.shape[1])
                logger.debug("Sampling parameters set")

                # Initialize tracking and sampling classes
                forest_container = ForestContainer(num_trees, 1, False)
                forest_sampler = ForestSampler(dataset, feature_types, num_trees, X.shape[0], alpha, beta, min_samples_leaf)
                cpp_rng = RNG(1234)  # Set a random seed
                global_var_model = GlobalVarianceModel()
                leaf_var_model = LeafVarianceModel()
                logger.debug("Tracking and sampling classes initialized")

                # Prepare to run the sampler
                num_warmstart = 10
                num_mcmc = 100
                num_samples = num_warmstart + num_mcmc
                global_var_samples = np.concatenate((np.array([global_variance_init]), np.repeat(0, num_samples)))
                logger.debug("Sampler prepared")

                try:
                    # Run the XBART sampler
                    logger.debug("Running XBART sampler")
                    for i in range(num_warmstart):
                        logger.debug(f"XBART iteration: {i}")
                        try:
                            forest_sampler.sample_one_iteration(forest_container, dataset, residual, cpp_rng, feature_types, cutpoint_grid_size, leaf_prior_scale, var_weights, global_var_samples[i], 1, True, False)
                            global_var_samples[i+1] = global_var_model.sample_one_iteration(residual, cpp_rng, nu, lamb)
                            logger.debug(f"XBART iteration {i} completed successfully")
                        except Exception as e:
                            logger.exception(f"Error during XBART iteration {i}:")
                            raise
                except Exception as e:
                    logger.exception("Error during XBART sampling:")
                    raise

                # Run the MCMC (BART) sampler
                logger.debug("Running MCMC (BART) sampler")
                for i in range(num_warmstart, num_samples):
                    try:
                        forest_sampler.sample_one_iteration(forest_container, dataset, residual, cpp_rng, feature_types, cutpoint_grid_size, leaf_prior_scale, var_weights, global_var_samples[i], 1, False, False)
                        global_var_samples[i+1] = global_var_model.sample_one_iteration(residual, cpp_rng, nu, lamb)
                        logger.debug(f"MCMC iteration {i - num_warmstart} completed successfully")
                    except Exception as e:
                        logger.exception(f"Error during MCMC iteration {i - num_warmstart}:")
                        raise

                # Extract mean function and error variance posterior samples
                forest_preds = forest_container.predict(dataset) * y_std + y_bar
                forest_preds_mcmc = forest_preds[:, num_warmstart:num_samples]
                logger.debug("Mean function and error variance posterior samples extracted")

                df_cleaned['Posterior Average (y hat)'] = forest_preds_mcmc.mean(axis=1)
                df_cleaned['2.5th percentile'] = np.percentile(forest_preds_mcmc, 2.5, axis=1)
                df_cleaned['97.5th percentile'] = np.percentile(forest_preds_mcmc, 97.5, axis=1)
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
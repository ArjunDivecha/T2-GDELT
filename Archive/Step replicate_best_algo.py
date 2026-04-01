"""
=======================================================================
STEP REPLICATE BEST ALGO - FACTOR TIMING STRATEGY REPLICATION
=======================================================================

PURPOSE:
--------
This script replicates the best performing factor timing algorithmic model (ElasticNet) 
from a comprehensive backtesting study and exports the out-of-sample portfolio weights 
to an Excel file for use in the T2 Optimizer system.

The script uses machine learning to predict which factors will perform well in the
future and creates portfolio weights accordingly.

INPUT FILES:
------------
1. T2_Optimizer.xlsx
   - Location: Same directory as this script
   - Purpose: Template file that defines the required column structure for output
   - Format: Excel file with factor names as columns and dates as rows
   - Required: Must exist before running this script

2. Data loaded by train.load_inputs() function:
   - Factor returns data (historical performance of different investment factors)
   - Macroeconomic indicators (economic data that affects factor performance)
   - Location: Loaded automatically by the train module
   - Format: Internal data format used by the training system

OUTPUT FILES:
-------------
1. output/Winner_15_Percent_Weights.xlsx
   - Location: output/ subdirectory (created automatically if needed)
   - Purpose: Contains the predicted portfolio weights for each factor over time
   - Format: Excel file with dates as rows and factor names as columns
   - Contents: Portfolio weights that sum to 1.0 (long-only constraint)
   - Usage: Can be imported into T2 Optimizer system for actual trading

VERSION HISTORY:
----------------
- Version 1.0: Initial implementation with ElasticNet model
- Last Updated: 2024
- Author: Quantitative Strategy Team

HOW IT WORKS (SIMPLIFIED):
---------------------------
1. Loads historical data about how different investment factors performed
2. Calculates technical indicators (momentum, volatility, etc.) for each factor
3. Trains an ElasticNet machine learning model to predict future performance
4. Uses expanding window approach - model learns from all available historical data
5. Generates predictions for which factors will do well next month
6. Creates portfolio weights focusing on top 10% of factors
7. Exports results in format ready for trading system

KEY CONCEPTS:
-------------
- FACTOR: An investment style (like value, momentum, size) that can be traded
- ELASTICNET: A type of machine learning model that prevents overfitting
- EXPANDING WINDOW: Training method that uses all available historical data
- LONG-ONLY: Strategy that only buys stocks, never shorts (sells)
- TOP 10%: Concentrating investments in best-performing factors only
======================================================================="""

# ====================================================================
# IMPORT LIBRARIES
# ====================================================================
# These are the tools we need to run our factor timing strategy:

import pandas as pd  # For working with data tables (like spreadsheets in code)
import numpy as np   # For mathematical calculations and random numbers
import os            # For creating directories
from sklearn.linear_model import ElasticNet  # The machine learning model we use
import warnings      # To hide warning messages and keep output clean
import train         # Our custom module with training and testing functions

# Hide warning messages to make the output easier to read
warnings.filterwarnings("ignore")

def replicate_and_export():
    """
    MAIN FUNCTION: REPLICATE THE BEST STRATEGY AND SAVE RESULTS
    =========================================================
    
    This is the main function that runs our complete factor timing strategy.
    Think of it as the recipe that follows all the steps from data preparation
    to final results.
    
    WHAT IT DOES (STEP BY STEP):
    ----------------------------
    
    STEP 1: GET READY
    - Set a random seed so we get the same results every time
    - Tell the system we want to focus on top 10% best factors only
    
    STEP 2: LOAD AND PREPARE DATA
    - Load historical factor performance data
    - Calculate technical indicators (momentum, volatility, etc.)
    - Set up the feature columns for our machine learning model
    
    STEP 3: SET UP THE MACHINE LEARNING MODEL
    - Configure the ElasticNet model with the best settings we found
    - These settings were determined by testing many different combinations
    
    STEP 4: TEST THE STRATEGY
    - Use expanding window: train on all available data, predict next month
    - Start with 60 months of data (minimum needed for reliable predictions)
    - Generate predictions for each month in our test period
    
    STEP 5: CHECK PERFORMANCE
    - Calculate how well our strategy would have performed
    - Show statistics like returns, risk-adjusted performance, maximum losses
    
    STEP 6: SAVE RESULTS
    - Format the predictions for the T2 Optimizer system
    - Save to Excel file in the exact format required for trading
    
    INPUTS NEEDED:
    --------------
    - T2_Optimizer.xlsx (template file)
    - Historical factor and macro data (loaded automatically)
    
    OUTPUTS CREATED:
    ---------------
    - output/Winner_15_Percent_Weights.xlsx (portfolio weights over time)
    - Performance statistics printed to screen
    
    RETURNS:
    --------
    None (results are saved to files and printed to screen)
    """
    # ====================================================================
    # STEP 1: INITIALIZATION - GETTING READY
    # ====================================================================
    # We need to set up our environment before we can start the analysis
    
    # Set random seed to 7 - this ensures we get the same results every time
    # Think of it like using the same recipe measurements every time you cook
    seed = 7
    np.random.seed(seed)
    
    # Tell the training system we want to use the "long_only_conc_p10" strategy
    # This means:
    # - long_only: We only buy factors, never sell them short
    # - conc_p10: We concentrate on the top 10% best-performing factors
    # This strategy was found to work best during our testing
    train.PRIMARY_PORTFOLIO = "long_only_conc_p10" 
    
    # ====================================================================
    # STEP 2: DATA LOADING AND FEATURE ENGINEERING
    # ====================================================================
    # Now we need to load our historical data and prepare it for analysis
    
    print("Loading datasets and engineering features...")
    
    # Load all the data we need:
    # - Historical factor returns (how each factor performed in the past)
    # - Macroeconomic data (economic indicators that affect markets)
    # - target_shift_months=0: We're predicting current month performance
    # - macro_lag_months=1: We use last month's economic data to predict this month
    panel = train.load_inputs(target_shift_months=0, macro_lag_months=1)
    
    # ====================================================================
    # STEP 3: FEATURE CONFIGURATION
    # ====================================================================
    # We need to tell our machine learning model which columns to use as features
    
    # Get the list of numeric feature columns from our data
    # These are the measurements our model will use to make predictions
    # Things like momentum, volatility, and other technical indicators
    train.FEATURE_COLUMNS = train.numeric_features(panel)
    
    # Set up aliases (different names for the same thing) that the model expects
    train.NUMERIC_FEATURES = train.FEATURE_COLUMNS
    
    # Get categorical features (if any exist) - these would be text-based categories
    train.CATEGORICAL_FEATURES = train.categorical_features()
    
    # ====================================================================
    # STEP 4: MODEL DEFINITION - SETTING UP OUR MACHINE LEARNING MODEL
    # ====================================================================
    # Now we define the exact machine learning model that performed best in our tests
    
    # Create our winning candidate with the optimal settings we discovered:
    winner = train.Candidate(
        name="elastic_net_v62",  # Name to identify this specific model version
        kind="fitted",           # Type of model (fitted means it learns from data)
        portfolio="long_only_conc_p10",  # The investment strategy we're using
        builder=lambda: train.make_linear_pipeline(
            # The ElasticNet model with the best hyperparameters:
            ElasticNet(
                alpha=0.060,        # Regularization strength (prevents overfitting)
                l1_ratio=0.35,      # Balance between two types of regularization
                max_iter=5000,      # Maximum iterations for the model to learn
                random_state=seed   # Use our random seed for reproducible results
            )
        )
    )
    
    # ====================================================================
    # STEP 5: WALK-FORWARD BACKTESTING - TESTING OUR STRATEGY
    # ====================================================================
    # Now we test our strategy using historical data to see how it would have performed
    
    print("Running Expanding Window Walk-Forward Backtest (this may take a minute) ...")
    
    # Run the backtest using the expanding window method:
    results = train.run_backtest(
        panel=panel,              # Our prepared data
        candidates=[winner],      # Just test our best model
        train_window=0,           # 0 = expanding window (use all available data)
        min_train_months=60,      # Need at least 60 months before making predictions
        top_k=12,                 # Select top 12 factors before final weighting
        n_jobs=-1                 # Use all available computer processors for speed
    )
    
    # WHAT EXPANDING WINDOW MEANS:
    # - Month 1-60: Train on first 60 months, predict month 61
    # - Month 1-61: Train on first 61 months, predict month 62
    # - Month 1-62: Train on first 62 months, predict month 63
    # - And so on... always using all available data to predict the next month
    
    # ====================================================================
    # STEP 6: PERFORMANCE EVALUATION - HOW WELL DID WE DO?
    # ====================================================================
    # Let's look at the results to see how our strategy performed
    
    # Get the performance statistics from our backtest
    leaderboard = results['leaderboard']
    
    # Display the results in a nice format
    print("\n" + "="*50)
    print(" FINAL ALGORITHM STATS (Out of Sample)")
    print("="*50)
    print(leaderboard.to_string(index=False))
    
    # WHAT THESE STATISTICS MEAN:
    # - Return: How much money the strategy made
    # - Sharpe Ratio: Return adjusted for risk (higher is better)
    # - Max Drawdown: Worst losing streak (lower is better)
    # - Volatility: How much the returns fluctuate
    
    # ====================================================================
    # STEP 7: EXPORT RESULTS FOR TRADING SYSTEM
    # ====================================================================
    # Now we save our results in the format that the T2 Optimizer system needs
    
    print("\nFormatting Time-Series Factor Weights matrix...")
    
    # Get the predictions for our winning model
    # This gives us the weights for each factor at each point in time
    best_preds = results['predictions'][train.candidate_key(winner)]
    
    # Convert dates to a format Excel can understand properly
    best_preds['Date'] = best_preds['Date'].dt.strftime('%Y-%m-%d')
    
    # Transform the data from long format to wide format:
    # FROM: Date, Factor, Weight (one row per factor per date)
    # TO:   Date as rows, Factors as columns, Weights as values
    pivoted = best_preds.pivot(index="Date", columns="factor", values="weight").fillna(0.0)
    
    # Make sure our columns match exactly what the T2 Optimizer expects
    t2_template = pd.read_excel('T2_Optimizer.xlsx')
    output_cols = [c for c in t2_template.columns if c != 'Date']
    final_output = pivoted.reindex(columns=output_cols, fill_value=0.0).reset_index()
    
    # Apply proper Excel date formatting according to the project guidelines
    output_path = 'output/Winner_15_Percent_Weights.xlsx'
    
    # Use xlsxwriter for proper date formatting (as required by project rules)
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Write the data
        final_output.to_excel(writer, sheet_name='Sheet1', index=False)
        
        # Get the workbook and worksheet to apply formatting
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # Create date format for the Date column
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        
        # Find the Date column index
        date_col_idx = final_output.columns.get_loc('Date')
        
        # Apply date formatting to all rows in the Date column
        for row in range(len(final_output)):
            worksheet.write(row + 1, date_col_idx, final_output['Date'].iloc[row], date_format)
        
        # Set column width for better visibility
        worksheet.set_column(date_col_idx, date_col_idx, 12)  # Width of 12 for dates
        
        # Set widths for factor columns
        for col_idx in range(len(final_output.columns)):
            if col_idx != date_col_idx:  # Skip the Date column
                worksheet.set_column(col_idx, col_idx, 15)  # Width of 15 for factor columns
    
    print(f"Successfully replicated and exported weights to '{output_path}'!")
    print("The file is ready for use in the T2 Optimizer system.")


# ====================================================================
# MAIN PROGRAM EXECUTION
# ====================================================================
# This is where the program actually starts running
# The code below only runs when you execute this file directly
# (not when you import it as a module into other programs)

if __name__ == '__main__':
    # Call our main function to run the complete strategy
    replicate_and_export()

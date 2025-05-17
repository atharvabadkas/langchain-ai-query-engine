import os
import numpy as np
import pandas as pd
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import sys
import subprocess
from datetime import datetime

# Install required packages
def install_requirements():
    required_packages = ['tabulate', 'langchain_experimental']
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            sys.exit(1)

# Function to validate and process the dataset
def validate_dataset(df):
    print("\nDataset Information:")
    print("-------------------")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    print("\nSample of the data:")
    print(df.head())
    
    # Check for date column
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) == 0:
        # Try to convert string date columns to datetime
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"\nConverted {col} to datetime format")
                except:
                    print(f"\nWarning: Could not convert {col} to datetime format")

    return df

def main():
    # Set up OpenAI API key
    api_key = os.environ.get('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')
    if not api_key:
        print("Error: OpenAI API key not found in environment variables")
        sys.exit(1)

    try:
        # Load data
        print("Loading dataset...")
        df = pd.read_csv("atharva-prep-dataset - master-data.csv")
        
        # Validate and process dataset
        df = validate_dataset(df)
        
        # Create Langchain agent
        print("\nInitializing Langchain agent...")
        agent = create_pandas_dataframe_agent(
            OpenAI(temperature=0),
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        # Example query
        print("\nExecuting query...")
        query = "What is the item weight of shira on 2nd Jan 2025?"
        print(f"Query: {query}")
        
        # Add error handling for the query execution
        try:
            result = agent.run(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            
        # Display additional data context
        print("\nRelevant data context:")
        if 'date' in df.columns:
            print(df[df['date'] == '2025-01-02'])
        else:
            print("Note: No 'date' column found in the dataset")
            
    except FileNotFoundError:
        print("Error: Dataset file not found")
    except pd.errors.EmptyDataError:
        print("Error: The dataset is empty")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    install_requirements()
    main()
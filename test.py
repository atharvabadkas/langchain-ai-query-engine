import os
os.environ['OPENAI_API_KEY'] = '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'  # Replace with your OpenAI API key

# Imports
import numpy as np
import pandas as pd
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import sys
import subprocess

# Install required packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tabulate'])

# Load data
df = pd.read_csv("atharva-prep-dataset - master-data.csv")

# Display first 3 rows (equivalent to df.head(3))
print("First 3 rows of the dataset:")
print(df.head(3))

# Create Langchain agent
agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0.5), 
    df, 
    verbose=True,
    allow_dangerous_code=True
)

# Example queries
def run_queries():
    # Query 1: Number of rows
    print("\nQuery 1: Find the item weight named 'shira' on the date 2025-01-02.")
    result = agent.run("Find the item weight named 'shira' on the date 2025-01-02.")
    print(result)

if __name__ == "__main__":
    run_queries()
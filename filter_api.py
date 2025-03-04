from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import uvicorn
import json
import pandas as pd
import dotenv
import io
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

# Define the tool for generating the pandas query
tools = [{
    "type": "function",
    "function": {
        "name": "generate_pandas_query",
        "description": (
            "Generate filter string to fetch user requested information using pandas (pd.DataFrame.query(expr)). "
            "Generate the 'expr' variable string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The filter string to be used in the pandas query."
                },
                "no_of_results": {
                    "type": "integer",
                    "description": "Number of results to be fetched."
                }
            },
            "required": ["query", "no_of_results"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

# Function to get the generated query from OpenAI
def get_result(user_query: str, df: pd.DataFrame):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        tools=tools,
        messages=[
            {"role": "system", "content": (
                "You are a resume ranking assistant. You will be provided a user query and a sample (head) "
                "of a dataframe. Your task is to generate a valid pandas query string that can be directly used "
                "in DataFrame.query(expr). For example, for a query 'Salary <= 100000 & Age < 40 & JOB.str.startswith(\"C\")', "
                "you need to return the filter string as 'Salary <= 100000 & Age < 40 & JOB.str.startswith(\"C\")' and also "
                "the number of results to be fetched (to be used with .head(x)). Do not use functions like .nlargest in your answer."
            )},
            {"role": "user", "content": f"user query: {user_query}\n\nHead of the dataframe:\n{df.head()}"}
        ]
    )
    # Extract tool call information from the response
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    generated_query = args['query']
    no_of_results = args['no_of_results']
    return generated_query, no_of_results

# Initialize FastAPI
app = FastAPI()

@app.post("/generate_query")
async def generate_query(
    query: str = Form(...),
    csv_file: UploadFile = File(...)
):
    # Parse the uploaded CSV into a Pandas DataFrame
    try:
        contents = await csv_file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")
    
    # Generate the pandas query string and number of results
    try:
        generated_query, no_of_results = get_result(query, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating query: {str(e)}")
    
    # Apply the generated query on the DataFrame and fetch results
    try:
        filtered_df = df.query(generated_query).head(no_of_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing query on dataframe: {str(e)}")
    
    # Return the generated query, number of results, and the filtered data
    return {
        "generated_query": generated_query,
        "no_of_results": no_of_results,
        "result": filtered_df.to_dict(orient="records")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

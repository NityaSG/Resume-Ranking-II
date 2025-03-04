from openai import OpenAI
import dotenv
import json
import pandas as pd
dotenv.load_dotenv()
client=OpenAI()
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "generate_pandas_query",
        "description": "Generate filter string to fetch user requested information using pandas (pd.Dataframe(exp)) generate the exp variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "the filter string of the variable to be used in the filtering"
                },
                "no_of_results": {
                    "type": "integer",
                    "description": "number of results to be fetched"
            }},
            "required": [
                "query","no_of_results"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    
# )

#print(completion.choices[0].message.tool_calls)
def get_result(query,df):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        tools=tools,
        messages=[
            {"role":"system","content":"""You are a resume ranking assistant. You will be provided a user query as an input and a head of a dataframe as a reference. You have to execute a pandas query to get the table for the user's question. please use the tool to execute, only the query which i can directly run in the pandas dataframe query. example : Salary  <= 100000 & Age < 40 & JOB.str.startswith("C").values.you have to generate the expr variable string for the following query command of pandas :  DataFrame.query(expr, *, inplace=False, **kwargs).head(x) just return the expr variable string. also return the number of results to be fetched which will be used in .head(x). also do not use .nlargest kind of things in the query variable"""},
            {"role":"user","content":f"user query : {query} \n\n head of the dataframe : {df.head()}"}
        ])
    #return response.choices[0].message.content.strip()
    tool_call=response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    query=args['query']
    x=args['no_of_results']
    #return response.choices[0].message.tool_calls
    return query,x

df=pd.read_csv("ml_job_candidates.csv")
query="3 candidates with highest experience in python"

result,x=get_result(query,df)
#result=str(result)
print(result)
print(df.query(result).head(x))

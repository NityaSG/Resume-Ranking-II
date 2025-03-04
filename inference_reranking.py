import io
import json
import csv
import os
import dotenv
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI  # Latest OpenAI client integration
from pinecone import Pinecone

dotenv.load_dotenv()
client = OpenAI()

# Initialize Pinecone client and index.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("INDEX_HOST"))

app = FastAPI(
    title="Resume Inference API",
    description=(
        "This API endpoint accepts a user question along with reference criteria and uses GPT to "
        "generate a semantic search query, metadata filter, and a candidate field for reranking. "
        "It then converts the query text to an embedding, performs a Pinecone vector query, optionally reranks "
        "the results, and returns a CSV with the candidate name, metadata, and rank."
    ),
    version="1.0"
)

@app.post("/infer-resumes", summary="Infer candidate matches with semantic search, metadata filtering, and reranking")
async def infer_resumes(
    question: str = Form(..., description="User question for semantic search."),
    criteria: str = Form(..., description="A JSON string representing reference criteria for generating filters.")
):
    # Parse the criteria JSON.
    try:
        criteria_data = json.loads(criteria)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid criteria format. Must be valid JSON.")

    # Updated function schema for GPT including a single rerank_field using enum.
    function_schema = {
        "name": "generate_semantic_search_query",
        "description": (
            "Generate a semantic search query text, metadata filter, and a candidate metadata field "
            "to use for reranking based on the user question and reference criteria. "
            "The query_text is used to generate a query embedding, and metadata_filter defines filter conditions for Pinecone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "The text query to be embedded for semantic search."
                },
                "metadata_filter": {
                    "type": "string",
                    "description": "A dictionary representing metadata filters. Keys are metadata field names and values are filter conditions."
                },
                "rerank_field": {
                    "type": "string",
                    "enum": ["experience_text", "projects_text", "skills_text"],
                    "description": "Candidate metadata field to use for reranking."
                }
            },
            "required": ["query_text", "metadata_filter", "rerank_field"]
        }
    }
    tool_definition = {
        "type": "function",
        "function": function_schema
    }
    system_message = {
        "role": "system",
        "content": (
            "You are an expert in semantic search. Generate a query, metadata filter, and a candidate metadata "
            "field for reranking based on the user's question and provided reference criteria. Use only one of the following fields "
            "for reranking: experience_text, projects_text, or skills_text. Also, ensure that total score field name is total_score."
        )
    }
    user_message = {
        "role": "user",
        "content": f"User question: {question}\nReference Criteria: {json.dumps(criteria_data, indent=2)}. Use these only for reference."
    }
    messages = [system_message, user_message]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            tools=[tool_definition],
            tool_choice="auto"
        )
        tool_call = completion.choices[0].message.tool_calls[0]
        semantic_query = json.loads(tool_call.function.arguments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating semantic search query: {e}")

    query_text = semantic_query.get("query_text")
    try:
        metadata_filter = json.loads(semantic_query.get("metadata_filter"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing metadata filter: {e}")
    rerank_field = semantic_query.get("rerank_field")

    # Create an embedding for the generated query_text.
    try:
        embedding_response = client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_vector = embedding_response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating query embedding: {e}")

    # Query the Pinecone index using the query vector and the generated metadata filter.
    try:
        pinecone_response = index.query(
            vector=query_vector,
            filter=metadata_filter,
            top_k=10,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {e}")

    matches = pinecone_response.get("matches", [])
    print("Matches received:", matches)
    print("Number of matches:", len(matches))

    # Perform reranking if a rerank field is provided.
    if rerank_field:
        print("Rerank field:", rerank_field)
        print("Matches received:", matches)
        documents = []
        for match in matches:
            # Use the default if the value is empty or evaluates to False.
            field_value = match["metadata"].get(rerank_field) or "waaheguru"
            print(f"Candidate {match['id']} - {rerank_field}: {field_value}")
            documents.append({
                "id": match["id"],
                rerank_field: field_value
            })
        print("Documents:", documents)
        try:
            rerank_result = pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query_text,
                documents=documents,
                rank_fields=[rerank_field],
                top_n=len(documents),
                return_documents=True,
                parameters={"truncate": "END"}
            )
            print("Rerank result:", rerank_result)
            # Reorder matches using the returned data (note the key "data" here)
            reranked_docs = rerank_result.get("data", [])
            new_order = []
            for rerank_doc in reranked_docs:
                # The document details are inside rerank_doc["document"]
                doc_id = rerank_doc["document"].get("id")
                match = next((m for m in matches if m["id"] == doc_id), None)
                if match:
                    new_order.append(match)
            matches = new_order
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in reranking: {e}")


    # Build CSV output with columns: Candidate Name, Metadata, and Rank.
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Candidate Name", "Metadata", "Rank"])
    for i, match in enumerate(matches, start=1):
        candidate_name = match["metadata"].get("candidate_name", "Unknown")
        metadata_str = json.dumps(match["metadata"])
        writer.writerow([candidate_name, metadata_str, i])
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

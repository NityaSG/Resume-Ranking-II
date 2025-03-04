import io
import csv
import json
import asyncio
import uuid
import os
import PyPDF2
import docx
import dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI  # Latest OpenAI client integration
from pinecone import Pinecone  # Pinecone client

dotenv.load_dotenv()
client = OpenAI()

# Initialize Pinecone client and index.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("INDEX_HOST"))

app = FastAPI(
    title="Combined Resume Processing API with Pinecone Upsert",
    description=(
        "Processes resumes by scoring them based on nested criteria and extracting resume details "
        "using GPT. It then creates embeddings from the education, experience, and skills fields and "
        "upserts them into a Pinecone vector database along with flattened score metadata and additional text fields."
    ),
    version="1.0"
)

# --- Utility functions to extract text from files ---
def extract_text_from_pdf(uploaded_file: UploadFile) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file.file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {e}")
    finally:
        uploaded_file.file.seek(0)
    return text

def extract_text_from_docx(uploaded_file: UploadFile) -> str:
    try:
        document = docx.Document(uploaded_file.file)
        text = "\n".join(para.text for para in document.paragraphs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DOCX: {e}")
    finally:
        uploaded_file.file.seek(0)
    return text

# --- GPT Function for Scoring a Resume ---
def get_gpt_scores(candidate_name: str, resume_text: str, criteria_data: dict) -> dict:
    """
    Uses GPT function calling to score a candidate's resume based on provided criteria.
    """
    function_schema = {
        "name": "score_resume",
        "description": (
            "Score a candidate's resume based on provided criteria. For each criterion in 'Must have', "
            "assign a score between 0 and 10; for 'Good to have', assign a score between 0 and 5; and for "
            "'Nice to have', assign a score between 0 and 2. Compute the total score as the sum of all individual scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_name": {
                    "type": "string",
                    "description": "Name of the candidate."
                },
                "scores": {
                    "type": "object",
                    "properties": {
                        "Must have": {
                            "type": "object",
                            "description": "Scores for 'Must have' criteria (values between 0 and 10).",
                            "additionalProperties": {"type": "number"}
                        },
                        "Good to have": {
                            "type": "object",
                            "description": "Scores for 'Good to have' criteria (values between 0 and 5).",
                            "additionalProperties": {"type": "number"}
                        },
                        "Nice to have": {
                            "type": "object",
                            "description": "Scores for 'Nice to have' criteria (values between 0 and 2).",
                            "additionalProperties": {"type": "number"}
                        }
                    },
                    "required": ["Must have", "Good to have", "Nice to have"],
                    "additionalProperties": False
                },
                "total_score": {
                    "type": "number",
                    "description": "The sum of all individual criterion scores."
                }
            },
            "required": ["candidate_name", "scores", "total_score"],
            "additionalProperties": False
        }
    }

    tool_definition = {
        "type": "function",
        "function": function_schema
    }

    system_message = {
        "role": "system",
        "content": "You are an expert resume evaluator. Use structured function calling to score candidates."
    }
    user_message = {
        "role": "user",
        "content": (
            f"Candidate Name: {candidate_name}\n\n"
            f"Resume Text:\n{resume_text}\n\n"
            "Criteria (nested JSON):\n"
            f"{json.dumps(criteria_data, indent=2)}\n\n"
            "Instructions:\n"
            "For each criterion group, assign scores using the following ranges:\n"
            "- Must have: 0 to 10\n"
            "- Good to have: 0 to 5\n"
            "- Nice to have: 0 to 2\n\n"
            "Calculate the total score as the sum of all individual scores. "
            "Return the result as a JSON object matching the schema of the function definition."
        )
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
        args = json.loads(tool_call.function.arguments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling GPT API for scoring: {e}")
    
    return args

# --- GPT Function for Extracting Resume Details ---
def get_extraction_details(resume_text: str, extraction_properties: dict) -> dict:
    """
    Uses GPT function calling to extract resume details based on a provided extraction schema.
    """
    tool_definition = {
        "type": "function",
        "function": {
            "name": "extract_resume_details",
            "description": "Extract paragraphs or values from the resume text for various attributes.",
            "parameters": {
                "type": "object",
                "properties": extraction_properties,
                "required": list(extraction_properties.keys()),
                "additionalProperties": False
            },
            "strict": True
        }
    }
    messages = [{
        "role": "user",
        "content": (
            "Extract the following details from the resume text below: "
            "Return each as a separate value as per their type.\n\n" + resume_text
        )
    }]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[tool_definition],
        )
        tool_call = completion.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling GPT API for extraction: {e}")
    
    return args

# --- Default Extraction Schema ---
default_extraction_properties = {
    "education": {
        "type": "string",
        "description": "Paragraph on education background."
    },
    "experience": {
        "type": "string",
        "description": "Paragraph on work experience."
    },
    "projects": {
        "type": "string",
        "description": "Paragraph on projects."
    },
    "skills": {
        "type": "string",
        "description": "Paragraph on skills."
    }
}

# --- FastAPI Endpoint ---
@app.post("/process-resumes", summary="Process Resumes: Score, Extract, and Upsert to Pinecone")
async def process_resumes(
    criteria: str = Form(
        ...,
        description=(
            "A JSON string representing nested criteria groups. Example format:\n"
            '{ "criteria": { "Must have": { "experience in AI application development": true, ... }, ... } }'
        )
    ),
    extraction_schema: str = Form(
        None,
        description=(
            "Optional JSON string representing additional extraction fields. "
            "It should have the key 'additional_fields' with a list of objects containing "
            "name, description, and datatype (e.g., string, number, boolean)."
        )
    ),
    files: list[UploadFile] = File(
        ...,
        description="List of resume files (PDF or DOCX)."
    )
):
    # Parse the criteria JSON
    try:
        criteria_payload = json.loads(criteria)
        criteria_data = criteria_payload.get("criteria")
        if not isinstance(criteria_data, dict):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid criteria format. Must be a JSON with a 'criteria' key containing nested groups.")

    # Build the extraction properties by starting with the default fields
    extraction_properties = default_extraction_properties.copy()
    if extraction_schema:
        try:
            extraction_schema_payload = json.loads(extraction_schema)
            additional_fields = extraction_schema_payload.get("additional_fields", [])
            for field in additional_fields:
                if "name" in field and field["name"].strip():
                    extraction_properties[field["name"].strip()] = {
                        "type": field.get("datatype", "string"),
                        "description": field.get("description", "")
                    }
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid extraction_schema format. Must be valid JSON.")

    results = []
    # Process each file
    for uploaded_file in files:
        filename = uploaded_file.filename
        candidate_name = filename.rsplit(".", 1)[0]
        if filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        elif filename.lower().endswith(".docx"):
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            continue  # Skip unsupported file types

        # Concurrently call GPT for scoring and extraction
        try:
            scoring_future = asyncio.to_thread(get_gpt_scores, candidate_name, resume_text, criteria_data)
            extraction_future = asyncio.to_thread(get_extraction_details, resume_text, extraction_properties)
            scoring_result, extraction_result = await asyncio.gather(scoring_future, extraction_future)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file '{filename}': {e}")

        # --- Create a unique candidate ID ---
        candidate_id = str(uuid.uuid4())

        # --- Prepare text for embedding: concatenate education, experience, and skills ---
        education_text = extraction_result.get("education", "")
        experience_text = extraction_result.get("experience", "")
        skills_text = extraction_result.get("skills", "")
        embedding_input = f"Education: {education_text}\nExperience: {experience_text}\nSkills: {skills_text}"

        # --- Create an embedding using OpenAI text-embedding-3-small ---
        try:
            embedding_response = client.embeddings.create(
                input=embedding_input,
                model="text-embedding-3-small"
            )
            embedding = embedding_response.data[0].embedding
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating embedding for candidate '{candidate_name}': {e}")

        # --- Flatten score dictionaries and prepare metadata ---
        metadata = {
            "candidate_name": candidate_name,
            "total_score": scoring_result.get("total_score", 0),
            # Add full text for experience, projects, and skills
            "experience_text": extraction_result.get("experience", ""),
            "projects_text": extraction_result.get("projects", ""),
            "skills_text": extraction_result.get("skills", "")
        }

        score_groups = scoring_result.get("scores", {})
        for group_name, group_scores in score_groups.items():
            # Create a prefix by converting the group name to lowercase and replacing spaces with underscores
            prefix = group_name.lower().replace(" ", "_")
            for key, value in group_scores.items():
                metadata[f"{prefix}_{key}"] = value

        # --- Upsert the vector to Pinecone with the flattened metadata ---
        try:
            index.upsert(
                vectors=[
                    {
                        "id": candidate_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error upserting to Pinecone for candidate '{candidate_name}': {e}")

        results.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "scores": scoring_result,
            "extraction": extraction_result
        })

    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

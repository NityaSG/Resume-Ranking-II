# Resume Ranking II 

## Task 1 : NL2Pandas Query based candidate shortlisting

![Flow_diagram](https://github.com/user-attachments/assets/586e6977-f5c0-4276-87bd-626192bc5ea9)


filter_api.py 
# Filter API

## Overview
The `Filter API` is a FastAPI-based application that allows users to upload a CSV file containing job candidate data and submit a natural language query to filter relevant candidates. The application leverages OpenAI's GPT-4o to generate a valid pandas query string based on the user's query and then applies it to the uploaded CSV file to return the relevant results.

## How It Works
1. A user submits a query (e.g., "3 candidates with highest experience in Python") along with a CSV file containing candidate data.
2. The API processes the CSV file and converts it into a Pandas DataFrame.
3. OpenAI's GPT-4o generates a valid Pandas query expression to filter the relevant candidates.
4. The generated query is applied to the DataFrame.
5. The filtered results are returned as JSON.

## Endpoints
### `POST /generate_query`
#### Request Parameters:
- **query** (string, required): A natural language query describing the filter criteria.
- **csv_file** (file, required): A CSV file containing the data to be filtered.

#### Request Example:
```bash
curl --location 'http://localhost:8000/generate_query' \
--form 'query="3 candidates with highest experience in python"' \
--form 'csv_file=@"/C:/Users/KIIT/Downloads/Resume_Ranking/ml_job_candidates.csv"'
```

#### Response Example:
```json
{
    "generated_query": "`Must have experience in python` == `Must have experience in python`.max()",
    "no_of_results": 3,
    "result": [
        {
            "Candidate Name": "JohnDoe",
            "Must have experience in python": 10,
            "Must have experience in machine learning": 10,
            "Experience with cloud platforms": 7,
            "Experience with deep learning": 8,
            "Experience with NLP": 9,
            "Experience with data engineering": 9,
            "Experience with MLOps": 5,
            "Experience with OpenAI": 0,
            "Total Score": 58
        },
        {
            "Candidate Name": "AlexJohnson",
            "Must have experience in python": 10,
            "Must have experience in machine learning": 10,
            "Experience with cloud platforms": 7,
            "Experience with deep learning": 4,
            "Experience with NLP": 4,
            "Experience with data engineering": 10,
            "Experience with MLOps": 6,
            "Experience with OpenAI": 4,
            "Total Score": 55
        },
        {
            "Candidate Name": "SophiaDavis",
            "Must have experience in python": 10,
            "Must have experience in machine learning": 8,
            "Experience with cloud platforms": 9,
            "Experience with deep learning": 8,
            "Experience with NLP": 5,
            "Experience with data engineering": 10,
            "Experience with MLOps": 4,
            "Experience with OpenAI": 1,
            "Total Score": 55
        }
    ]
}
```

## Installation and Running
### Requirements
- Python 3.8+
- `pip install fastapi uvicorn openai pandas python-dotenv`

### Running the API
```bash
uvicorn filter_api:app --host 0.0.0.0 --port 8000
```

## Notes
- Ensure your OpenAI API key is configured in your environment variables before running the API.
- The CSV file should have appropriate column headers relevant to the query being processed.
- The model is configured to return valid Pandas `DataFrame.query(expr)` expressions that are directly executable.



2. upload.py : 

# Resume Processing API

## Overview
The `Resume Processing API` is a FastAPI-based application designed to:
1. Process resumes in PDF or DOCX format.
2. Extract relevant candidate details using OpenAI's GPT-4o.
3. Score candidates based on predefined criteria.
4. Generate text embeddings for education, experience, and skills fields.
5. Upsert the processed data into a Pinecone vector database.

## How It Works
1. A user submits multiple resume files and provides scoring criteria in JSON format.
2. The API extracts the text from each resume (PDF or DOCX).
3. OpenAI GPT-4o generates a structured scoring report and extracts candidate details.
4. The extracted details (education, experience, projects, skills) are embedded into a vector using OpenAI's `text-embedding-3-small` model.
5. The vector and metadata are stored in Pinecone for future retrieval.

## Endpoints
### `POST /process-resumes`
#### Request Parameters:
- **criteria** (string, required): A JSON string specifying scoring criteria. Example:
  ```json
  {
    "criteria": {
      "Must have": {
        "experience in python": true,
        "experience in machine learning": true
      },
      "Good to have": {
        "cloud certification": true
      },
      "Nice to have": {
        "experience with openai": true
      }
    }
  }
  ```
- **extraction_schema** (string, optional): A JSON string defining additional extraction fields. Example:
  ```json
  {
    "additional_fields": [
      {"name": "certifications", "description": "List of professional certifications", "datatype": "string"},
      {"name": "hobbies", "description": "Hobbies and interests", "datatype": "string"}
    ]
  }
  ```
- **files** (list[UploadFile], required): A list of resume files (PDF or DOCX).

#### Request Example:
```bash
curl --location 'http://localhost:8000/process-resumes' \
--form 'criteria="{
    \"criteria\": {
      \"Must have\": {\"experience in python\": true, \"experience in machine learning\": true},
      \"Good to have\": {\"cloud certification\": true},
      \"Nice to have\": {\"experience with openai\": true}
    }
}"' \
--form 'extraction_schema="{
    \"additional_fields\": [{\"name\": \"certifications\", \"description\": \"List of professional certifications\", \"datatype\": \"string\"}]
}"' \
--form 'files=@"/path/to/resume.pdf"'
```

#### Response Example:
```json
{
    "results": [
        {
            "candidate_id": "dc32ac72-8bfc-4acd-8fe1-1d78d702c7da",
            "candidate_name": "Updated_Resume__Sanjulika_MLE_or_DS (1)",
            "scores": {
                "candidate_name": "Sanjulika Sharma",
                "scores": {
                    "Must have": {
                        "experience in python": 10,
                        "experience in machine learning": 10
                    },
                    "Good to have": {
                        "cloud certification": 0
                    },
                    "Nice to have": {
                        "experience with openai": 2
                    }
                },
                "total_score": 22
            },
            "extraction": {
                "education": "Master of Science in Data Science",
                "experience": "Worked at GeeksForGeeks as a Data Scientist",
                "projects": "Developed an AI-powered chatbot",
                "skills": "Python, Machine Learning, NLP",
                "certifications": "AWS Certified Machine Learning"
            }
        }
    ]
}
```

## Installation and Running
### Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install fastapi uvicorn openai pandas python-dotenv pinecone-client PyPDF2 python-docx
  ```

### Running the API
```bash
uvicorn resume_processing_api:app --host 0.0.0.0 --port 8000
```

## Notes
- Ensure your OpenAI and Pinecone API keys are set in environment variables before running the API.
- The API supports batch processing of multiple resumes at once.
- Extracted resume details and scores are stored in Pinecone for retrieval.

3. Inference.py : 
# Resume Inference API with Reranking

## Overview
The `Resume Inference API` is a FastAPI-based application designed to:
1. Accept a user query along with reference criteria.
2. Use OpenAI's GPT-4o to generate a semantic search query, metadata filter, and a reranking field.
3. Convert the query into an embedding.
4. Query Pinecone for candidate matches.
5. Optionally rerank the results based on a selected metadata field.
6. Return a CSV file containing matched candidates along with metadata and rank.

## How It Works
1. A user submits a query (e.g., "I need candidates with an overall score greater than 20") and reference criteria.
2. OpenAI's GPT-4o generates a structured search query, metadata filter, and a reranking field.
3. The API creates an embedding for the generated search query.
4. Pinecone is queried using the embedding and metadata filter.
5. If a reranking field (e.g., `experience_text`, `projects_text`, `skills_text`) is selected, the results are reranked using the `bge-reranker-v2-m3` model.
6. The results are compiled into a CSV file and returned to the user.

## Endpoints
### `POST /infer-resumes`
#### Request Parameters:
- **question** (string, required): A user question defining the search criteria.
- **criteria** (string, required): A JSON string specifying filtering criteria. Example:
  ```json
  {
    "criteria": {
      "Must have": {
        "experience in python": true,
        "experience in machine learning": true
      },
      "Good to have": {
        "cloud certification": true
      },
      "Nice to have": {
        "experience with openai": true
      }
    }
  }
  ```

#### Request Example:
```bash
curl --location 'http://localhost:8000/infer-resumes' \
--form 'question="I need candidates with an overall score greater than 20"' \
--form 'criteria="{
  \"criteria\": {
    \"Must have\": {
      \"experience in python\": true,
      \"experience in machine learning\": true
    },
    \"Good to have\": {
      \"cloud certification\": true
    },
    \"Nice to have\": {
      \"experience with openai\": true
    }
  }
}"'
```

#### Response Example (CSV format):
```
Candidate Name,Metadata,Rank
JohnDoe,"{\"candidate_name\": \"JohnDoe\", \"total_score\": 27.0, \"experience_text\": \"Worked on AI projects...\", \"projects_text\": \"Developed NLP models...\", \"skills_text\": \"Python, TensorFlow, NLP...\"}",1
JaneDoe,"{\"candidate_name\": \"JaneDoe\", \"total_score\": 25.0, \"experience_text\": \"Worked on ML projects...\", \"projects_text\": \"Built predictive models...\", \"skills_text\": \"Python, Scikit-learn, Data Engineering...\"}",2
```

## Installation and Running
### Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install fastapi uvicorn openai python-dotenv pinecone-client
  ```

### Running the API
```bash
uvicorn resume_inference_api:app --host 0.0.0.0 --port 8000
```

## Notes
- Ensure your OpenAI and Pinecone API keys are set in environment variables before running the API.
- The API supports filtering by metadata fields and ranking candidates based on Pinecone similarity scores.
- If reranking is enabled, candidates are re-ranked based on the selected field (`experience_text`, `projects_text`, or `skills_text`).
- The response is a downloadable CSV file containing matched candidates.

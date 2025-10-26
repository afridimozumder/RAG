from fastapi import FastAPI
from pydantic import BaseModel
from llm import generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    answer = generate_answer(req.question)
    return {"answer": answer}

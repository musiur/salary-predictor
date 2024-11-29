from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import atexit
import logging
import json

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Define the request model to accept JSON body
class PromptRequest(BaseModel):
    prompt: str  # Accepting the resume text as a string

app = FastAPI()
ollama_model_chat = "llama3.2"  # Model identifier for the chat

# Global variable to store the Ollama process
ollama_process = None

def start_ollama():
    """Starts Ollama in the background."""
    global ollama_process
    try:
        logging.info("Starting Ollama server...")
        ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("Ollama server started.")
    except Exception as e:
        logging.error(f"Failed to start Ollama: {str(e)}")

def stop_ollama():
    """Stops the Ollama process when the app exits."""
    global ollama_process
    if ollama_process:
        ollama_process.terminate()
        ollama_process.wait()
        logging.info("Ollama server stopped.")

# Register the cleanup function
atexit.register(stop_ollama)

# Start Ollama when the server runs
start_ollama()


@app.post("/generate/")
async def generate_prompt(request: PromptRequest):
    try:
        # Extract the resume data from the request body
        prompt = request.prompt

        # Construct the custom prompt to extract specific details
        custom_prompt = f"""
Please extract the following details from the resume data provided below and return them in the exact format specified:
1. "age" - Return a number type (integer). If age is not mentioned, return 26.
2. "gender" - Return either "Male" or "Female". If not mentioned, return "Male".
3. "education" - Return only the highest degree as "Bachelor's", "Master's", "PhD", or "High School". Remember to remove any degree levels like BSc, MSc, etc. and give the highest degree only like Bachelor's, Master's, PhD, High School as label and not more than one word.
4. "experience" - Calculate total years of experience by comparing the job start and end dates. Return a number type (integer).
5. "designation" - Return only the one designation.

Respond with only a JSON object with keys: "age", "gender", "education", "experience", and "designation". Lastly remember you don't need to add any extra words or explanation but only the extracted data in JSON format

Here is the resume data: {prompt}
"""


        # Log the custom prompt for debugging
        logging.debug(f"Generated custom prompt: {custom_prompt}")

        # Run the model using Ollama and pass input via stdin
        cmd = ["ollama", "run", ollama_model_chat]
        result = subprocess.run(cmd, input=custom_prompt, text=True, capture_output=True)

        # Check for errors in the result
        if result.returncode != 0:
            logging.error(f"Error from Ollama: {result.stderr.strip()}")
            raise HTTPException(status_code=500, detail=result.stderr.strip())

        # Return the model output as JSON
        return {"response": result.stdout.strip()}
        # return result.stdout.strip()

    except Exception as e:
        logging.error(f"Exception during prompt generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

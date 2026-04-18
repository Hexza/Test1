"""
Medical Report Analyzer - Web Application Backend
Run with: python app.py
Then open http://localhost:5000 in your browser
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
from flask_cors import CORS
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: Run -> pip install google-genai flask flask-cors")
    exit(1)

load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ParameterDetail(BaseModel):
    name: str
    extracted_value: str
    standard_range: str
    status: str          # 'Normal' | 'High' | 'Low'
    explanation: str
    implication: str

class ReportSummary(BaseModel):
    parameters: List[ParameterDetail]
    overall_health: str  # 'Healthy' | 'Needs attention' | 'Critical'
    key_findings: List[str]
    risk_indicators: List[str]
    suggested_next_steps: List[str]

# ── Analyzer ──────────────────────────────────────────────────────────────────

class MedicalReportAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing from .env file")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

    def system_instructions(self):
        return """
        You are a highly specialized medical data extraction architect.
        Analyze the provided pathology or blood test report.

        Rules:
        1. Extract ALL biomarkers: Hematology (Hb, RBC, WBC, Plt), Metabolic (Glucose, Cholesterol), Endocrine (TSH, T3, T4), and any others present.
        2. Use reference ranges from the document; fall back to standard clinical ranges if absent.
        3. Categorize each parameter strictly as 'Normal', 'High', or 'Low'.
        4. Give clear non-technical explanations and educational implications.
        5. Overall health MUST be exactly one of: 'Healthy', 'Needs attention', 'Critical'.
        6. NEVER diagnose. All output is educational and informational only.
        """

    def analyze(self, file_bytes: bytes, filename: str) -> dict:
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        uploaded_file = None
        try:
            uploaded_file = self.client.files.upload(file=tmp_path)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    "Analyze this medical report and extract all biomarkers.",
                    uploaded_file
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instructions(),
                    response_mime_type="application/json",
                    response_schema=ReportSummary,
                    temperature=0.0,
                )
            )
            data = json.loads(response.text)
            return ReportSummary(**data).model_dump()
        finally:
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except:
                    pass
            os.unlink(tmp_path)

analyzer = MedicalReportAnalyzer()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    allowed = {".pdf", ".png", ".jpg", ".jpeg"}
    if Path(f.filename).suffix.lower() not in allowed:
        return jsonify({"error": "Unsupported file type. Use PDF, PNG, or JPG."}), 400
    try:
        result = analyzer.analyze(f.read(), f.filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🩺  Medical Report Analyzer is running!")
    print("👉  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)

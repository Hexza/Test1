"""
Intelligent Multimodal Medical Report Analysis System
Dependencies: google-genai, pydantic, python-dotenv, click, pytesseract, Pillow
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

import click
from dotenv import load_dotenv

# Optional OCR imports for explicit fallback mechanisms
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Import the modern Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("CRITICAL ERROR: The 'google-genai' package is not installed.")
    print("Execute: pip install google-genai")
    sys.exit(1)


# ==========================================
# 1. Structured Output Schemas (Pydantic)
# ==========================================

class ParameterDetail(BaseModel):
    name: str = Field(description="Name of the physiological biomarker (e.g., Hemoglobin, TSH, WBC Count)")
    extracted_value: str = Field(description="The precise numerical value and unit extracted from the document")
    standard_range: str = Field(description="The reference range (prioritize document ranges; fallback to standard clinical ranges)")
    status: str = Field(description="Strict evaluation. Must be exactly one of: 'Normal', 'High', 'Low'")
    explanation: str = Field(description="Clear, non-technical explanation of the parameter's physiological origin")
    implication: str = Field(description="Educational health implications of this specific value. Strictly non-diagnostic.")

class ReportSummary(BaseModel):
    parameters: List[ParameterDetail] = Field(description="Comprehensive array of all extracted biomarkers")
    overall_health: str = Field(description="Strict categorization. Must be one of: 'Healthy', 'Needs attention', 'Critical'")
    key_findings: List[str] = Field(description="2-3 bulleted observations summarizing the metabolic or hematologic state")
    risk_indicators: List[str] = Field(description="Specific risk factors identified from deviations. Empty list if none exist.")
    suggested_next_steps: List[str] = Field(description="Actionable lifestyle modifications or explicit referrals to medical professionals")


# ==========================================
# 2. Core AI Processing Engine
# ==========================================

class MedicalReportAnalyzer:
    """Encapsulates the Gemini API interactions, multimodal processing, and OCR fallbacks."""

    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Environment configuration failure: GEMINI_API_KEY is missing.")

        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

    def get_system_instructions(self) -> str:
        """Defines the rigid behavioral and ethical constraints for the LLM."""
        return """
        You are a highly specialized medical data extraction architect. Your directive is to meticulously analyze the provided pathology or blood test report.

        Rigid Operational Constraints:
        1. Extract all available biomarkers, prioritizing Hematology (Hb, RBC, WBC, Plt), Metabolic (Glucose, Cholesterol), and Endocrine (TSH, T3, T4).
        2. Evaluate the extracted metrics against the specific reference ranges printed on the document. If absent, apply standard international clinical ranges.
        3. Categorize every parameter strictly as 'Normal', 'High', or 'Low'.
        4. Provide an educational explanation of the parameter's physiological mechanism and the general implications of its current level.
        5. Evaluate the macroscopic health condition STRICTLY as 'Healthy', 'Needs attention', or 'Critical'. Assign 'Critical' if severe deviations (e.g., critical anemia, severe leukocytosis) are detected.
        6. ETHICAL MANDATE: DO NOT formulate a medical diagnosis. All explanations must remain informative and explicitly non-diagnostic.
        """

    def extract_text_via_ocr(self, image_path: Path) -> str:
        """Fallback mechanism: Uses Tesseract OCR to extract text if forced or required."""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR requested but 'pytesseract' or 'Pillow' is not installed.")
        print(" Initiating local Tesseract OCR text extraction pipeline...")
        try:
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)
            return extracted_text
        except Exception as e:
            raise RuntimeError(f"OCR Pipeline Failure: {str(e)}")

    def process_document(self, file_path: str, force_ocr: bool = False) -> ReportSummary:
        """Orchestrates file uploading, API invocation, and JSON deserialization."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File system error: {file_path} cannot be located.")

        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.txt']
        if path.suffix.lower() not in supported_extensions:
            raise ValueError(f"Incompatible file architecture. Supported formats: {supported_extensions}")

        print(f"\n Establishing secure connection. Uploading {path.name}...")

        # FIX: contents_payload must include the analysis prompt
        contents_payload = [
            "Please analyze this medical report and extract all biomarkers according to your instructions."
        ]

        uploaded_file = None

        try:
            if force_ocr and path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                raw_text = self.extract_text_via_ocr(path)
                contents_payload.append(f"OCR Extracted Text:\n{raw_text}")
            else:
                uploaded_file = self.client.files.upload(file=str(path))
                contents_payload.append(uploaded_file)

            print(" Executing multimodal analysis via Gemini 2.5 Flash. Awaiting response...")

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents_payload,
                config=types.GenerateContentConfig(
                    system_instruction=self.get_system_instructions(),
                    response_mime_type="application/json",
                    response_schema=ReportSummary,
                    temperature=0.0,
                )
            )

            json_data = json.loads(response.text)
            return ReportSummary(**json_data)

        except Exception as e:
            raise RuntimeError(f"AI Processing Architecture Failure: {str(e)}")
        finally:
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass


# ==========================================
# 3. CLI Application Interface (Click Framework)
# ==========================================

def print_disclaimer():
    """Renders the legally mandated medical AI disclaimer."""
    click.secho("\n" + "="*70, fg="red", bold=True)
    click.secho("!!! MANDATORY LEGAL AND MEDICAL DISCLAIMER!!!", fg="red", bold=True)
    click.echo("This analytical system utilizes Artificial Intelligence to parse data.")
    click.echo("The output is explicitly NOT medical advice, nor is it a medical diagnosis.")
    click.echo("Users are strictly advised to consult a qualified healthcare professional")
    click.echo("prior to making any clinical or health-related decisions.")
    click.secho("="*70 + "\n", fg="red", bold=True)

def format_terminal_output(report: ReportSummary):
    """Translates the Pydantic data model into a highly readable CLI topology."""

    click.secho("\n" + "#"*60, fg="blue", bold=True)

    health_color = (
        "green" if report.overall_health.lower() == "healthy"
        else ("yellow" if report.overall_health.lower() == "needs attention"
              else "red")
    )
    click.secho(f" MACROSCOPIC HEALTH EVALUATION: {report.overall_health.upper()}", fg=health_color, bold=True)
    click.secho("#"*60 + "\n", fg="blue", bold=True)

    click.secho("--- EXTRACTED PHYSIOLOGICAL BIOMARKERS ---", bold=True)
    for param in report.parameters:
        if param.status.lower() == "normal":
            status_flag = click.style("✔ NORMAL", fg="green")
        elif param.status.lower() == "high":
            status_flag = click.style("[↑ HIGH]", fg="yellow")
        else:
            status_flag = click.style("[↓ LOW]", fg="red")

        click.echo(f"\n> {click.style(param.name, bold=True)} {status_flag}")
        click.echo(f"  Extracted Value : {param.extracted_value} (Standard Interval: {param.standard_range})")
        click.echo(f"  Origin/Mechanism: {param.explanation}")
        click.echo(f"  Clinical Impact : {param.implication}")

    click.secho("\n--- MACRO OBSERVATIONS & KEY FINDINGS ---", bold=True)
    for finding in report.key_findings:
        click.echo(f"  • {finding}")

    if report.risk_indicators:
        click.secho("\n--- IDENTIFIED RISK VECTORS ---", fg="red", bold=True)
        for risk in report.risk_indicators:
            click.echo(f" ⚠ {risk}")

    click.secho("\n--- ACTIONABLE RECOMMENDATIONS ---", fg="cyan", bold=True)
    for step in report.suggested_next_steps:
        click.echo(f"  ➔ {step}")

    click.echo("\n" + "="*70)
    click.echo("END OF ANALYTICAL REPORT")
    click.echo("="*70 + "\n")

@click.command()
@click.option('--file', prompt='Enter the exact file path of the medical document', help='Path to the PDF, PNG, or JPG report.')
@click.option('--force-ocr', is_flag=True, help='Bypass native vision and force local Tesseract OCR text extraction.')
def main(file, force_ocr):
    """Intelligent Multimodal CLI Analyzer for Medical Pathology Reports."""

    click.clear()
    click.secho("="*70, fg="blue", bold=True)
    click.secho("  INTELLIGENT MEDICAL REPORT ANALYZER (Powered by Gemini AI)  ", fg="blue", bold=True)
    click.secho("="*70, fg="blue", bold=True)

    print_disclaimer()

    clean_path = file.strip("\"'")

    try:
        analyzer = MedicalReportAnalyzer()
        report_data = analyzer.process_document(clean_path, force_ocr=force_ocr)
        format_terminal_output(report_data)
        print_disclaimer()
    except Exception as e:
        click.secho(f"\n⚠ A critical error occurred during execution:\n{str(e)}\n", fg="red", bold=True)

if __name__ == "__main__":
    main()

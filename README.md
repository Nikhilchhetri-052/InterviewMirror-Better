# InterviewMirror

InterviewMirror is an AI-assisted mock interview web application that helps candidates practice role-based interviews with adaptive follow-up questions, answer evaluation, and speech-enabled interaction.

## Demo

[![Watch Demo](https://img.youtube.com/vi/vjetB-Eohpg/0.jpg)]
(https://youtu.be/vjetB-Eohpg)
## Purpose
- Help learners and job seekers prepare for technical and role-specific interviews.
- Simulate realistic interview flow with main and follow-up questions.
- Provide actionable feedback on technical quality, communication, and depth.
- Support admins with job posting and interview-session tracking.

## Core Features
- User authentication and role-based access (`interviewee`, `admin`).
- Mock interview sessions with:
  - Dynamic question generation.
  - Follow-up question generation.
  - Structured answer scoring and feedback.
- Speech-to-text transcription via Whisper (`/record`).
- Text-to-speech response playback via `pyttsx3` (`/speak`).
- Resume upload and parsing endpoint (`/upload_resume`).
- Admin panel for job listing create/edit/deactivate.
- User interview history and session persistence in MongoDB.

## Tech Stack
- Backend: Python, Flask, Flask-CORS
- Database: MongoDB, Flask-PyMongo
- Auth/Security: Flask-Bcrypt, Flask sessions
- AI/LLM:
  - Ollama local API (default text generation/evaluation path)
  - Fallback question bank and evaluation paths in code
- Speech/Audio:
  - OpenAI Whisper (`openai-whisper` + `torch`) for transcription
  - `pyttsx3` for TTS
  - `soundfile` for audio handling
- NLP/Resume Parsing: spaCy, `pdfplumber`, `docx2txt`, `sentence-transformers`
- Frontend: Jinja2 templates, HTML/CSS/JS

## Project Structure
- `app.py`: Main Flask app, routes, auth, admin and interview APIs.
- `interviewgenerate.py`: Interview session lifecycle and summary logic.
- `llm_adapter.py`: LLM orchestration (generation, follow-up, scoring).
- `ollama_client.py`: HTTP client for local Ollama generation API.
- `question_bank.py`, `data/*.json`: Role-wise question data.
- `resume_parser.py`: Resume parsing utility used by upload endpoint.
- `templates/`: Jinja templates.
- `static/`: CSS, scripts, and assets.
- `interview_models/`: Local model artifacts and evaluation assets.

## Installation
1. Clone the repository and move into the project folder.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure MongoDB is running locally on default port (`27017`) unless you change configuration.
5. Ensure Ollama is installed and running locally if you want LLM-powered generation.

## Configuration
Create a `.env` file for local environment values as needed.

Current code defaults include:
- Mongo URI in `app.py`: `mongodb://localhost:27017/InterviewMirror`
- Ollama endpoint in `ollama_client.py`: `http://localhost:11434/api/generate`
- Optional LLM model env var used by `llm_adapter.py`:
  - `OLLAMA_MODEL` (default: `mistral`)

Recommended improvement: move hardcoded `MONGO_URI` and Flask `secret_key` into environment variables for production.

## Running the Application
```bash
python app.py
```

Default local URL:
- `http://127.0.0.1:5000/`

## Key Endpoints
- Web routes:
  - `/`, `/login`, `/register`, `/logout`
  - `/interview`, `/interviewafterlogin`
  - `/admin_dashboard`, `/admin_setting`, `/admin/jobs`
- APIs:
  - `POST /api/interview/start`
  - `POST /api/interview/answer`
  - `POST /api/interview/end`
  - `POST /api/generate_feedback`
  - `POST /record`
  - `POST /speak`
  - `POST /upload_resume`

## Data and Models
- Large model files (`.h5`, `model.safetensors`) are stored in the repository.
- Interview role datasets are provided in `data/*.json`.
- Runtime artifacts (for example, transcriptions) may be appended to local files such as `transcriptions.txt`.

## Future Enhancements
- Move all sensitive settings to environment variables and add config profiles (dev/staging/prod).
- Add automated tests (unit + API integration) and CI pipeline.
- Add Docker support for reproducible local/dev deployment.
- Add robust logging and monitoring for model latency and API errors.
- Persist interview analytics dashboards (progress over time, skill-gap trends).
- Improve resume parsing accuracy for varied formats and multilingual resumes.
- Add JWT/session hardening and rate-limiting for public endpoints.
- Add async/background processing for long-running inference tasks.
- Improve UX for interview mode switching (voice-only, text-only, hybrid).

## Known Limitations
- Some configuration values are currently hardcoded in source.
- Production-grade security/deployment settings are not fully wired.
- Local model/runtime setup (Ollama, Whisper, Torch) can be resource-intensive.

## Contributing
1. Create a feature branch.
2. Keep changes focused and documented.
3. Run local validation before opening a PR.
4. Submit a PR with a clear summary of behavior changes.


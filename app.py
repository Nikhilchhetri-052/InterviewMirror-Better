from ollama_client import ollama_generate
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify,send_file
from dotenv import load_dotenv
from bson.objectid import ObjectId
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import uuid
import whisper
import pyttsx3
import tempfile
import soundfile as sf
from flask_cors import CORS
from interviewgenerate import manager, set_database_collections
from llm_adapter import llm_service
from datetime import datetime
import re

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/InterviewMirror" #for deploy app.config["MONGO_URI"] = "mongodb+srv://Raunak:gemsschool123@cluster0.yeh8tgu.mongodb.net/InterviewMirror" #"mongodb://localhost:27017/InterviewMirror"
app.config["UPLOAD_FOLDER"] = "static/uploads"
mongo = PyMongo(app)
app.secret_key = "9b1c8c49d3c3e7e456f7ab97d8332a19"
bcrypt = Bcrypt(app)
users_collection = mongo.db.users
jobs_collection = mongo.db.jobs
interview_sessions_collection = mongo.db.interview_sessions
job_applications_collection = mongo.db.job_applications

# Initialize database collections in interview manager
set_database_collections(interview_sessions_collection, job_applications_collection)

model = whisper.load_model("base")


def parse_custom_questions(raw_value):
    if not raw_value:
        return []

    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    lines = str(raw_value).splitlines()
    questions = []
    for line in lines:
        cleaned = re.sub(r'^\s*(?:[-*]|\d+[\).\:-])\s*', '', line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions

    
@app.context_processor
def inject_user():
    return dict(
        logged_in_user=session.get('username'),
        logged_in_user_id=session.get('custom_user_id')
    )
@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = users_collection.find_one({"email": email})
        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']  
            session['custom_user_id'] = user['user_id']
            session['role'] = user.get('role', 'interviewee')  # Store role in session
            flash('Login successful!', 'success')
            
            # Redirect based on role
            if user.get('role') == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('homepageafterlogin'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user_role = request.form.get('userRole', 'interviewee')  # Default to interviewee

        # Add validation for empty fields
        if not username or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return redirect(url_for('register'))

        if not password.strip():  # Check for non-empty after trimming
            raise ValueError("Password must be non-empty.")

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        if users_collection.find_one({"email": email}):
            flash("Email already registered!", "warning")
            return redirect(url_for('register'))

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        # 👇 Generate next user ID
        last_user = users_collection.find_one(sort=[("user_id", -1)])
        if last_user and last_user.get("user_id", "").startswith("USER"):
            # Extract numeric part & increment
            last_id_num = int(last_user["user_id"][4:])
            new_id_num = last_id_num + 1
        else:
            new_id_num = 1

        user_id = f"USER{new_id_num:03d}"  # e.g., USER001, USER002, ...

        user = {
            "user_id": user_id,   # shorter readable ID
            "username": username,
            "email": email,
            "password": hashed_pw,
            "role": user_role  # Add role to user document
        }

        users_collection.insert_one(user)

        flash(f"Registration successful! Your User ID is {user_id}. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('custom_user_id', None)
    session.pop('role', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

@app.route('/homepageafterlogin', methods=['GET', 'POST'])
def homepageafterlogin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('homepageafterlogin.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/aboutafterlogin')
def aboutafterlogin():
    return render_template('aboutafterlogin.html')
@app.route('/Contactus')
def Contactus():
    return render_template('Contactus.html')

@app.route('/Contactafterlogin')
def Contactafterlogin():
    return render_template('Contactafterlogin.html')

@app.route('/job_apply')
def job_apply():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    jobs = list(jobs_collection.find({"active": True}).sort("created_at", -1))
    for job in jobs:
        job['_id'] = str(job['_id'])
    return render_template('job_apply.html', jobs=jobs)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('homepageafterlogin'))
    
    # Get all active job listings
    jobs = list(jobs_collection.find({"active": True}).sort("created_at", -1))
    for job in jobs:
        job['_id'] = str(job['_id'])
    
    return render_template('admin_dashborad.html', jobs=jobs)

@app.route('/admin_setting')
def admin_setting():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('homepageafterlogin'))
    
    # Get all jobs posted by this admin
    admin_user_id = session.get('custom_user_id')
    admin_jobs = list(jobs_collection.find({"created_by": admin_user_id}).sort("created_at", -1))
    for job in admin_jobs:
        job['_id'] = str(job['_id'])

    # Show candidates who applied to this admin's jobs and have at least one completed interview
    admin_job_ids = [job['_id'] for job in admin_jobs]
    admin_applications = list(
        job_applications_collection.find({"created_by": admin_user_id}).sort("applied_at", -1)
    )

    candidates_by_user = {}
    for application in admin_applications:
        candidate_user_id = application.get("candidate_user_id")
        if not candidate_user_id:
            continue

        if candidate_user_id not in candidates_by_user:
            user_doc = users_collection.find_one(
                {"user_id": candidate_user_id, "role": {"$ne": "admin"}}
            )
            if not user_doc:
                continue
            user_doc['_id'] = str(user_doc['_id'])
            user_doc['interview_sessions'] = []
            user_doc['applications'] = []
            candidates_by_user[candidate_user_id] = user_doc

        app_id = application.get("_id")
        all_app_sessions = []
        if app_id is not None:
            all_app_sessions = list(
                interview_sessions_collection.find({"application_id": str(app_id)}).sort("last_updated_at", -1)
            )
        if not all_app_sessions and application.get("latest_interview_id"):
            all_app_sessions = list(
                interview_sessions_collection.find(
                    {"interview_id": application.get("latest_interview_id")}
                )
            )

        completed_sessions = [session_doc for session_doc in all_app_sessions if session_doc.get("status") == "completed"]
        if not completed_sessions:
            continue

        for session_doc in completed_sessions:
            session_doc['_id'] = str(session_doc['_id'])
            candidates_by_user[candidate_user_id]['interview_sessions'].append(session_doc)

        candidates_by_user[candidate_user_id]['applications'].append(
            {
                "job_title": application.get("job_title", "Unknown Job"),
                "job_company": application.get("job_company", ""),
                "status": application.get("interview_status", "unknown"),
                "applied_at": application.get("applied_at"),
                "interview_completed_at": application.get("interview_completed_at"),
            }
        )

    # Backward compatibility: include completed sessions tied directly to admin job ids
    if admin_job_ids:
        legacy_sessions = list(
            interview_sessions_collection.find(
                {"job_id": {"$in": admin_job_ids}, "status": "completed"}
            ).sort("completed_at", -1)
        )
        for session_doc in legacy_sessions:
            candidate_user_id = session_doc.get("user_id")
            if not candidate_user_id:
                continue

            if candidate_user_id not in candidates_by_user:
                user_doc = users_collection.find_one(
                    {"user_id": candidate_user_id, "role": {"$ne": "admin"}}
                )
                if not user_doc:
                    continue
                user_doc['_id'] = str(user_doc['_id'])
                user_doc['interview_sessions'] = []
                user_doc['applications'] = []
                candidates_by_user[candidate_user_id] = user_doc

            if any(existing.get("interview_id") == session_doc.get("interview_id") for existing in candidates_by_user[candidate_user_id]['interview_sessions']):
                continue

            session_doc['_id'] = str(session_doc['_id'])
            candidates_by_user[candidate_user_id]['interview_sessions'].append(session_doc)

    def session_sort_key(session_doc):
        return (
            session_doc.get("completed_at")
            or session_doc.get("last_updated_at")
            or session_doc.get("created_at")
            or ""
        )

    for candidate in candidates_by_user.values():
        sessions = candidate.get("interview_sessions", [])
        if not sessions:
            continue

        sessions.sort(key=session_sort_key, reverse=True)
        latest_session = sessions[0]
        latest_summary = latest_session.get("summary") or {}
        average_scores = latest_summary.get("average_scores") or {}
        overall_score = float(average_scores.get("overall", 0) or 0)

        if overall_score >= 4.0:
            recommendation = "Strongly Recommended"
        elif overall_score >= 3.0:
            recommendation = "Recommended"
        elif overall_score >= 2.0:
            recommendation = "Needs Review"
        else:
            recommendation = "Not Recommended"

        candidate["latest_performance"] = {
            "overall": round(overall_score, 2),
            "technical": average_scores.get("technical", "N/A"),
            "communication": average_scores.get("communication", "N/A"),
            "depth": average_scores.get("depth", "N/A"),
            "questions_completed": latest_summary.get("questions_completed", 0),
            "total_questions": latest_summary.get("total_questions", 0),
            "summary_text": latest_summary.get("summary", "No summary available."),
            "recommendation": recommendation,
            "completed_at": latest_session.get("completed_at") or latest_session.get("last_updated_at"),
            "role": latest_session.get("role") or latest_summary.get("role"),
        }

    candidates = list(candidates_by_user.values())
    
    return render_template('admin_setting.html', candidates=candidates, jobs=admin_jobs)

@app.route('/api/job/apply', methods=['POST'])
def apply_for_job():
    if 'user_id' not in session:
        return jsonify({"error": "Login required"}), 401

    data = request.get_json() or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    try:
        job = jobs_collection.find_one({"_id": ObjectId(job_id), "active": True})
    except Exception:
        return jsonify({"error": "Invalid job_id"}), 400

    if not job:
        return jsonify({"error": "Job not found"}), 404

    application_filter = {
        "job_id": job_id,
        "candidate_user_id": session.get("custom_user_id"),
    }
    existing_application = job_applications_collection.find_one(application_filter)

    if existing_application:
        application_id = str(existing_application["_id"])
    else:
        application = {
            "job_id": job_id,
            "job_title": job.get("title"),
            "job_company": job.get("company"),
            "job_role": job.get("title") or job.get("job_type") or "general",
            "created_by": job.get("created_by"),
            "candidate_user_id": session.get("custom_user_id"),
            "candidate_username": session.get("username"),
            "applied_at": datetime.utcnow().isoformat(),
            "interview_status": "not_started",
            "interview_started_at": None,
            "interview_completed_at": None,
            "latest_interview_id": None,
        }
        result = job_applications_collection.insert_one(application)
        application_id = str(result.inserted_id)

    redirect_url = url_for(
        'interviewafterlogin',
        role=(job.get("title") or job.get("job_type") or "general"),
        job_id=job_id,
        application_id=application_id,
    )
    return jsonify({"success": True, "application_id": application_id, "redirect_url": redirect_url})

# Job Management Routes
@app.route('/admin/jobs', methods=['GET', 'POST'])
def admin_jobs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('homepageafterlogin'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        company = request.form.get('company')
        location = request.form.get('location')
        workplace = request.form.get('workplace')
        job_type = request.form.get('job_type')
        description = request.form.get('description')
        question_source = request.form.get('question_source', 'developer')
        custom_questions = request.form.get('custom_questions')
        parsed_custom_questions = parse_custom_questions(custom_questions)

        if question_source == 'custom' and not parsed_custom_questions:
            flash('Please provide at least one custom question when custom source is selected.', 'danger')
            return redirect(url_for('admin_jobs'))
        
        job = {
            "title": title,
            "company": company,
            "location": location,
            "workplace": workplace,
            "job_type": job_type,
            "description": description,
            "question_source": question_source,
            "custom_questions": "\n".join(parsed_custom_questions),
            "created_by": session.get('custom_user_id'),
            "created_at": datetime.utcnow(),
            "active": True
        }
        
        jobs_collection.insert_one(job)
        flash('Job listing created successfully!', 'success')
        return redirect(url_for('admin_jobs'))
    
    # Get all jobs (both active and inactive) for management view
    jobs = list(jobs_collection.find().sort("created_at", -1))
    for job in jobs:
        job['_id'] = str(job['_id'])
    return render_template('admin_jobs.html', jobs=jobs)

@app.route('/admin/jobs/<job_id>/edit', methods=['GET', 'POST'])
def edit_job(job_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('homepageafterlogin'))
    
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        flash('Job not found.', 'danger')
        return redirect(url_for('admin_jobs'))
    
    if request.method == 'POST':
        question_source = request.form.get('question_source', 'developer')
        custom_questions = request.form.get('custom_questions')
        parsed_custom_questions = parse_custom_questions(custom_questions)
        if question_source == 'custom' and not parsed_custom_questions:
            flash('Please provide at least one custom question when custom source is selected.', 'danger')
            return redirect(url_for('edit_job', job_id=job_id))

        update_data = {
            "title": request.form.get('title'),
            "company": request.form.get('company'),
            "location": request.form.get('location'),
            "workplace": request.form.get('workplace'),
            "job_type": request.form.get('job_type'),
            "description": request.form.get('description'),
            "question_source": question_source,
            "custom_questions": "\n".join(parsed_custom_questions),
        }
        
        jobs_collection.update_one({"_id": ObjectId(job_id)}, {"$set": update_data})
        flash('Job updated successfully!', 'success')
        return redirect(url_for('admin_jobs'))
    
    # Convert _id to string for template
    job['_id'] = str(job['_id'])
    return render_template('edit_job.html', job=job)

@app.route('/admin/jobs/<job_id>/delete')
def delete_job(job_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('homepageafterlogin'))
    
    jobs_collection.update_one({"_id": ObjectId(job_id)}, {"$set": {"active": False}})
    flash('Job deactivated successfully!', 'success')
    return redirect(url_for('admin_jobs'))

@app.route('/interview')
def interview():
    role = request.args.get('role', 'frontend')  # default to 'frontend'
    return render_template('interview.html', role=role)

@app.route('/interviewafterlogin')
def interviewafterlogin():
    role = request.args.get('role', 'frontend')  # default to 'frontend'
    return render_template('interviewafterlogin.html', role=role)

@app.route('/userhistory', methods=['GET', 'POST'])
def userhistory():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect if not logged in

    user_id = ObjectId(session['user_id'])
    user = users_collection.find_one({"_id": user_id})
    interview_sessions = list(
        interview_sessions_collection.find({"user_id": session.get('custom_user_id')}).sort("last_updated_at", -1)
    )
    for interview_session in interview_sessions:
        interview_session['_id'] = str(interview_session['_id'])

    return render_template('userhistory.html', user=user, interview_sessions=interview_sessions)

@app.route('/record', methods=['POST'])
def record():
    import whisper, tempfile
    from flask import request, jsonify

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']

    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp:
        audio_file.save(temp.name)
        temp.flush()

        # Load whisper model (cached globally)
        global model
        result = model.transcribe(temp.name)
        text = result['text']

    # Optionally save transcription
    with open("transcriptions.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

    return jsonify({'text': text})
@app.route("/api/generate_feedback", methods=["POST"])
def generate_feedback():
    data = request.get_json()
    answer = data.get("answer", "")
    question = data.get("question", "")
    role = data.get("role", "")

    feedback_text = llm_service.generate_feedback(role, question, answer)

    return jsonify({"feedback": feedback_text})


@app.route("/api/interview/start", methods=["POST"])
def interview_start():
    data = request.get_json() or {}
    role = data.get("role", "general")
    difficulty = data.get("difficulty", "medium")
    job_id = data.get("job_id")
    application_id = data.get("application_id")
    interview_id = data.get("interview_id")
    custom_questions = []
    try:
        max_questions = int(data.get("max_questions", 5))
    except (TypeError, ValueError):
        max_questions = 5

    # Get user_id if logged in
    user_id = session.get('custom_user_id') if 'custom_user_id' in session else None

    if application_id:
        try:
            application_doc = job_applications_collection.find_one({"_id": ObjectId(application_id)})
        except Exception:
            application_doc = None
        if application_doc:
            job_id = application_doc.get("job_id", job_id)

    job_doc = None
    if job_id:
        try:
            job_doc = jobs_collection.find_one({"_id": ObjectId(job_id)})
        except Exception:
            job_doc = None

    if job_doc:
        if not data.get("role"):
            role = job_doc.get("title") or role
        if job_doc.get("question_source") == "custom":
            custom_questions = parse_custom_questions(job_doc.get("custom_questions"))

    session_obj = manager.start_session(
        role=role,
        difficulty=difficulty,
        max_questions=max_questions,
        user_id=user_id,
        job_id=job_id,
        application_id=application_id,
        custom_questions=custom_questions,
        resume_interview_id=interview_id,
    )

    return jsonify(
        {
            "interview_id": session_obj.interview_id,
            "question": session_obj.current_question,
            "mode": session_obj.current_mode,
            "question_index": session_obj.current_index + 1,
            "total_questions": session_obj.max_questions,
        }
    )


@app.route("/api/interview/answer", methods=["POST"])
def interview_answer():
    data = request.get_json() or {}
    interview_id = data.get("interview_id")
    answer = data.get("answer", "")

    if not interview_id:
        return jsonify({"error": "Missing interview_id"}), 400
    if not answer:
        return jsonify({"error": "Missing answer"}), 400

    result = manager.submit_answer(interview_id, answer)
    return jsonify(result)


@app.route("/api/interview/end", methods=["POST"])
def interview_end():
    data = request.get_json() or {}
    interview_id = data.get("interview_id")
    if not interview_id:
        return jsonify({"error": "Missing interview_id"}), 400

    result = manager.end_session(interview_id)
    return jsonify(result)

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Create a temporary file
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")

        # Choose a voice (female if available)
        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)

        # Save to audio file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        # Send the file to the frontend
        return send_file(temp_path, mimetype="audio/wav", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # pyttsx3 creates file before runAndWait; don't delete until after it's sent
        pass

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    file = request.files['resume']
    save_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
    file.save(save_path)

    from resume_parser import parse_resume
    data = parse_resume(save_path)

    return jsonify({"parsed_resume": data})



if __name__ == '__main__':
    
    app.run(debug=True)

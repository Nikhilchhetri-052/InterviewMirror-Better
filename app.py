from flask import Flask, render_template, request, redirect, url_for, session, flash
from bson.objectid import ObjectId
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import uuid
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/InterviewMirror"
app.config["UPLOAD_FOLDER"] = "static/uploads"
mongo = PyMongo(app)
app.secret_key = "9b1c8c49d3c3e7e456f7ab97d8332a19"
bcrypt = Bcrypt(app)
users_collection = mongo.db.users

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
            session['username'] = user['username']   # 👈 store username
            session['custom_user_id'] = user['user_id']  # 👈 store short USER001 id
            flash('Login successful!', 'success')
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
            "password": hashed_pw
        }

        users_collection.insert_one(user)

        flash(f"Registration successful! Your User ID is {user_id}. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
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

    return render_template('userhistory.html', user=user)

if __name__ == '__main__':
    
    app.run(debug=True)
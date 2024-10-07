from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import TruncatedSVD
import numpy as np
import torch
from werkzeug.utils import secure_filename
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from docx import Document
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'

user_profiles = pd.DataFrame([
    {'id': 1, 'name': 'Alice', 'skills': '3D printing, art design', 'goals': 'create innovative 3D art', 'hobbies': 'painting, sculpting', 'password': 'password123', 'user_type': 'youth'},
    {'id': 2, 'name': 'Bob', 'skills': 'networking, IT support', 'goals': 'enhance cybersecurity skills', 'hobbies': 'gaming, problem solving', 'password': 'password456', 'user_type': 'adult'},
    {'id': 3, 'name': 'Carol', 'skills': 'photography, digital media', 'goals': 'mentor youth in digital projects', 'hobbies': 'photography, film making', 'password': 'password789', 'user_type': 'instructor'},
    {'id': 4, 'name': 'David', 'skills': 'software development, AI', 'goals': 'build an AI-based system', 'hobbies': 'coding, gaming', 'password': 'password101', 'user_type': 'youth'},
    {'id': 5, 'name': 'Eve', 'skills': 'data science, AI development', 'goals': 'use AI for social good', 'hobbies': 'machine learning, research', 'password': 'password102', 'user_type': 'instructor'}
])

#  Projects data
projects = pd.DataFrame([
    {
        'id': 101, 
        'title': 'AI-Powered Art Generation Tool', 
        'description': 'Develop a tool that leverages artificial intelligence to generate unique pieces of digital art. The project involves AI model training, image processing, and UI/UX design to enable users to create their own artwork with minimal input.', 
        'skills_required': 'AI development, machine learning, UI/UX design, image processing'
    },
    {
        'id': 102, 
        'title': 'Cybersecurity Incident Response System', 
        'description': 'Design and implement a real-time cybersecurity incident response system. This system will help monitor, detect, and respond to security threats in real time, providing detailed analytics and automated responses to prevent breaches.', 
        'skills_required': 'cybersecurity, networking, software development, data analysis'
    },
    {
        'id': 103, 
        'title': 'Youth Mentorship Platform', 
        'description': 'Create a web-based platform connecting mentors with youth looking to develop skills in technology, arts, or business. The platform will include mentorship matching algorithms, chat features, and progress tracking tools.', 
        'skills_required': 'web development, database management, user experience, mentoring'
    },
    {
        'id': 104, 
        'title': 'Sustainable Energy Monitoring System', 
        'description': 'Develop an IoT-based monitoring system that tracks and optimizes energy consumption for households and businesses using renewable energy sources. The project includes hardware integration, data analytics, and cloud-based reporting.', 
        'skills_required': 'IoT, data analytics, cloud computing, renewable energy systems'
    },
    {
        'id': 105, 
        'title': '3D Printed Prosthetics for Low-Income Communities', 
        'description': 'Design and produce cost-effective, 3D-printed prosthetics for individuals in low-income communities. This project focuses on creating customizable, durable prosthetic limbs using 3D printing technology.', 
        'skills_required': '3D printing, biomedical engineering, CAD design, material science'
    },
    {
        'id': 106, 
        'title': 'Virtual Reality Education Modules', 
        'description': 'Build interactive virtual reality modules for education in subjects like history, science, and engineering. The project will involve VR content creation, instructional design, and testing with students and educators.', 
        'skills_required': 'virtual reality development, instructional design, programming, education technology'
    },
    {
        'id': 107, 
        'title': 'AI-Driven Healthcare Diagnostics', 
        'description': 'Develop a machine learning model to assist in healthcare diagnostics, focusing on image recognition for detecting diseases in medical scans.', 
        'skills_required': 'AI development, medical imaging, healthcare data, deep learning'
    },
    {
        'id': 108, 
        'title': 'Blockchain for Secure Voting', 
        'description': 'Build a blockchain-based voting platform to ensure transparency, security, and anonymity in elections.', 
        'skills_required': 'blockchain development, cryptography, web development'
    }
])

# Events data
events = pd.DataFrame([
    {'id': 201, 'title': 'Art Workshop', 'description': 'A workshop on the basics of art and creativity.', 'skills_required': 'art, creativity'},
    {'id': 202, 'title': 'Cybersecurity Conference', 'description': 'A conference on cybersecurity and network security.', 'skills_required': 'networking, cybersecurity'},
    {'id': 203, 'title': 'Photography Exhibition', 'description': 'An exhibition showcasing photographs by young talents.', 'skills_required': 'photography, digital media'},
    {'id': 204, 'title': 'IT Networking Bootcamp', 'description': 'A bootcamp for learning and practicing IT networking skills.', 'skills_required': 'networking, IT support'},
    {'id': 205, 'title': 'AI Symposium', 'description': 'A symposium where industry experts and researchers discuss recent advancements in artificial intelligence.', 'skills_required': 'AI development, machine learning'},
    {'id': 206, 'title': 'Sustainable Energy Summit', 'description': 'An event focused on innovative solutions in sustainable energy and green technologies.', 'skills_required': 'renewable energy, green technologies'}
])

# user-specific projects and events
user_projects = {
    1: [101, 103],  # Project IDs
    2: [102, 104],
    3: [103, 104],
    4: [105, 107],
    5: [106, 108]
}

# Event IDs
user_events = {
    1: [201, 202],  
    2: [202, 204],
    3: [203, 204],
    4: [205, 206],
    5: [205, 206]
}

# Expanded interactions data
interactions = pd.DataFrame([
    # Project interactions
    {'user_id': 1, 'item_id': 101, 'interaction': 'completed', 'type': 'project'},
    {'user_id': 1, 'item_id': 103, 'interaction': 'liked', 'type': 'project'},
    {'user_id': 2, 'item_id': 102, 'interaction': 'completed', 'type': 'project'},
    {'user_id': 2, 'item_id': 104, 'interaction': 'clicked', 'type': 'project'},
    {'user_id': 3, 'item_id': 103, 'interaction': 'completed', 'type': 'project'},
    {'user_id': 3, 'item_id': 104, 'interaction': 'liked', 'type': 'project'},
    {'user_id': 4, 'item_id': 105, 'interaction': 'completed', 'type': 'project'},
    {'user_id': 4, 'item_id': 107, 'interaction': 'liked', 'type': 'project'},
    {'user_id': 5, 'item_id': 106, 'interaction': 'completed', 'type': 'project'},
    {'user_id': 5, 'item_id': 108, 'interaction': 'clicked', 'type': 'project'},

    # Event interactions
    {'user_id': 1, 'item_id': 201, 'interaction': 'completed', 'type': 'event'},
    {'user_id': 1, 'item_id': 202, 'interaction': 'liked', 'type': 'event'},
    {'user_id': 2, 'item_id': 203, 'interaction': 'clicked', 'type': 'event'},
    {'user_id': 3, 'item_id': 204, 'interaction': 'completed', 'type': 'event'},
    {'user_id': 2, 'item_id': 201, 'interaction': 'liked', 'type': 'event'},
    {'user_id': 4, 'item_id': 205, 'interaction': 'completed', 'type': 'event'},
    {'user_id': 5, 'item_id': 205, 'interaction': 'liked', 'type': 'event'},
    {'user_id': 5, 'item_id': 206, 'interaction': 'completed', 'type': 'event'}
])


# Interaction mapping for different interaction types
interaction_map = {'completed': 1, 'liked': 0.5, 'clicked': 0.2}
interactions['interaction'] = interactions['interaction'].map(interaction_map)

# Create interaction matrix for SVD
interaction_matrix = interactions.pivot_table(index='user_id', columns='item_id', values='interaction', fill_value=0)

# perform SVD to get user and project factors
svd = TruncatedSVD(n_components=2)
user_factors = svd.fit_transform(interaction_matrix)
item_factors = svd.components_.T

# generated predicted interactions using dot product of user and project factors
predicted_interactions = np.dot(user_factors, item_factors.T)
predicted_df = pd.DataFrame(predicted_interactions, index=interaction_matrix.index, columns=interaction_matrix.columns)

# Load BERT model for content-based recommendations
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# combine user skills, goals, and hobbies into a single profile text
user_profiles['profile_text'] = user_profiles['skills'] + '. ' + user_profiles['goals'] + '. ' + user_profiles['hobbies']
projects['project_text'] = projects['description'] + '. ' + projects['skills_required']
events['event_text'] = events['description'] + '. ' + events['skills_required']

# generated embeddings for user profiles, projects, and events
user_embeddings = model.encode(user_profiles['profile_text'].tolist(), convert_to_tensor=True)
project_embeddings = model.encode(projects['project_text'].tolist(), convert_to_tensor=True)
event_embeddings = model.encode(events['event_text'].tolist(), convert_to_tensor=True)

# calculated content similarities using cosine similarity
project_similarities = util.pytorch_cos_sim(user_embeddings, project_embeddings)
event_similarities = util.pytorch_cos_sim(user_embeddings, event_embeddings)

# generate hybrid recommendations for projects
def hybrid_recommendations(user_id, top_n=2):
    # fetch user index in the profiles DataFrame
    user_index = user_profiles.index[user_profiles['id'] == user_id].tolist()[0]
    
    # recommendations using cosine similarity
    user_similarities = project_similarities[user_index]
    top_project_indices = torch.topk(user_similarities, k=top_n).indices
    content_based_recommendations = projects.iloc[top_project_indices.tolist()]

    # collaborative filtering recommendations using SVD-predicted interactions
    if user_id in predicted_df.index:
        collaborative_recommendations = predicted_df.loc[user_id].sort_values(ascending=False).index[:top_n]
        collaborative_projects = projects[projects['id'].isin(collaborative_recommendations)]
    else:
        collaborative_projects = pd.DataFrame()  # If no collaborative data, return empty
    
    # content-based and collaborative filtering results, and remove duplicates
    final_recommendations = pd.concat([content_based_recommendations, collaborative_projects]).drop_duplicates(subset='id').head(top_n)
    
    return final_recommendations

#  generate hybrid recommendations for events
def hybrid_event_recommendations(user_id, top_n=2):
    user_index = user_profiles.index[user_profiles['id'] == user_id].tolist()[0]
    
    # recommendations using cosine similarity
    user_event_similarities = event_similarities[user_index]
    top_event_indices = torch.topk(user_event_similarities, k=top_n).indices
    content_based_event_recommendations = events.iloc[top_event_indices.tolist()]
    
    # content-based filtering only for events
    return content_based_event_recommendations

def get_user_projects_and_events(user_id):
    projects_list = projects[projects['id'].isin(user_projects.get(user_id, []))]  # Fetch from hardcoded data
    events_list = events[events['id'].isin(user_events.get(user_id, []))]  # Fetch from hardcoded data
    return projects_list, events_list

# Index Route (Main page for login or account creation)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user_id = session['user_id']
        # fetch the user data from user_profiles using the ID
        user = user_profiles[user_profiles['id'] == user_id].iloc[0]
        return render_template('dashboard.html', user=user)
    else:
        return redirect(url_for('login'))

# Route for My Projects
@app.route('/projects')
def projects_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    user_projects_list, _ = get_user_projects_and_events(user_id)
    return render_template('projects.html', projects=user_projects_list)

# Route for Suggested Projects
@app.route('/projects/suggested')
def suggested_projects_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    recommendations = hybrid_recommendations(user_id)  # Get hybrid recommendations for projects
    return render_template('suggested_projects.html', recommendations=recommendations)

# Route for My Events
@app.route('/events')
def events_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    _, user_events_list = get_user_projects_and_events(user_id)
    return render_template('events.html', events=user_events_list)

# Route for Suggested Events (using hybrid event recommendations)
@app.route('/events/suggested')
def suggested_events_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    event_recommendations = hybrid_event_recommendations(user_id)  # Get hybrid recommendations for events
    return render_template('suggested_events.html', recommendations=event_recommendations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])  # Access the user_id from the form
        password = request.form['password']
        
        user = user_profiles[user_profiles['id'] == user_id]

        if not user.empty and user.iloc[0]['password'] == password:
            session['user_id'] = user_id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid user ID or password')
            return redirect(url_for('login'))
    
    return render_template('login.html')

# Route for account creation
@app.route('/create-account', methods=['GET', 'POST'])
def create_account():
    global user_profiles

    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        skills = request.form['skills']
        goals = request.form['goals']
        hobbies = request.form['hobbies']

        new_user_id = user_profiles['id'].max() + 1

        new_user = pd.DataFrame([{
            'id': new_user_id,
            'name': name,
            'skills': skills,
            'goals': goals,
            'hobbies': hobbies
        }])

        user_profiles = pd.concat([user_profiles, new_user], ignore_index=True)

        session['user_id'] = new_user_id
        return redirect(url_for('dashboard'))

    return render_template('create_account.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

### Certificate Generation

# Google Drive authentication
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() 
    return GoogleDrive(gauth)

# Function to download the template from Google Drive
def download_template(drive, template_id):
    template_file = drive.CreateFile({'id': template_id})
    template_file.GetContentFile('template.docx', mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

# Function to upload certificates back to Google Drive
def upload_certificate(drive, name, folder_id, doc_bytes):
    cert_file = drive.CreateFile({'title': f'{name}_Certificate.docx', 'parents': [{'id': folder_id}]})
    cert_file.SetContentFile(f'{name}_Certificate.docx')
    cert_file.Upload()

# Route to handle file upload and certificate generation
@app.route('/upload', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file.save(filename)
        
        data = pd.read_excel(filename)
        print('data', data)

        # data.columns = data.columns.str.strip()

        # Google Drive authentication
        drive = authenticate_drive()

        template_id = "1Vp_4aKkWYXBnkq6ja7g2EdFru7v4h6jBpPB8dxizmPY" 
        download_template(drive, template_id)
        
        # Generate certificates for each name and upload to Google Drive
        folder_id = "1ib_sGypIWHU_2U3lz0f6q85w3MuRjXRF"
        generate_and_upload_certificates(data['Names'], drive, folder_id)
        # return redirect(url_for('certificate_success'))
        return render_template('certificate_success.html')
        # return 'Certificates generated and uploaded successfully!'

# Function to generate and upload certificates
def generate_and_upload_certificates(names, drive, folder_id):
    for name in names:
        # Create a new document for each certificate using the downloaded template
        doc = Document('template.docx')
        
        # Replace the placeholder with the actual name
        for paragraph in doc.paragraphs:
            for run in paragraph.runs:
                if '{NAME}' in run.text:
                    run.text = run.text.replace('{NAME}', name)
            
        doc.save(f'{name}_Certificate.docx')

        # Upload the certificate to Google Drive
        upload_certificate(drive, name, folder_id, f'{name}_Certificate.docx')


if __name__ == '__main__':
    app.run(debug=True)

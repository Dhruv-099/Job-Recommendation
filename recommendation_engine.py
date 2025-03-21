import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_roles(input_role, roles_data, top_n=3):
    roles_data['skills_str'] = roles_data['skills'].apply(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer()
    skills_matrix = vectorizer.fit_transform(roles_data['skills_str'])
    similarity_matrix = cosine_similarity(skills_matrix)
    idx = roles_data.index[roles_data['role'] == input_role].tolist()[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return roles_data.iloc[recommended_indices]['role'].tolist()
roles_data = pd.DataFrame({
    'role': ['Data Scientist', 'ML Engineer', 'Data Analyst', 'AI Researcher', 'Backend Engineer',
             'Frontend Engineer', 'Full Stack Engineer', 'Embedded Systems Engineer', 'Cloud Engineer',
             'Data Engineer', 'BI Analyst', 'Product Manager', 'DevOps Engineer', 'Cybersecurity Analyst',
             'Mobile App Developer', 'Game Developer'],
    'skills': [
        {'Python', 'Machine Learning', 'Statistics', 'SQL'},
        {'Python', 'Deep Learning', 'TensorFlow', 'PyTorch'},
        {'SQL', 'Excel', 'Data Visualization', 'Business Intelligence'},
        {'Deep Learning', 'Neural Networks', 'Research', 'Python'},
        {'Java', 'Spring Boot','Node.js', 'Microservices', 'MongoDB', 'Redis', 'Docker', 'Kafka'},
        {'JavaScript', 'React', 'CSS', 'HTML', 'Vue.js', 'TypeScript'},
        {'JavaScript', 'Node.js', 'React', 'MongoDB', 'Express', 'REST APIs', 'Docker'},
        {'C', 'C++', 'Microcontrollers', 'RTOS'},
        {'AWS', 'Azure', 'Cloud Security', 'Terraform', 'Kubernetes'},
        {'SQL', 'ETL', 'Data Warehousing', 'Big Data'},
        {'Power BI', 'SQL', 'Dashboarding', 'Data Analysis'},
        {'Product Strategy', 'Agile', 'Stakeholder Management', 'Data Analytics'},
        {'CI/CD', 'Kubernetes', 'Cloud Computing', 'Infrastructure as Code'},
        {'Network Security', 'Penetration Testing', 'Risk Management', 'Cybersecurity Fundamentals'},
        {'Swift', 'Kotlin', 'Flutter', 'Mobile UI/UX'},
        {'Unity', 'C#', 'Game Physics', '3D Rendering'}
    ]
})

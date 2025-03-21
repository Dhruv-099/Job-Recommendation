import pandas as pd
from sklearn.metrics import jaccard_score
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
def recommend_roles(input_role, roles_data, top_n=3):
    input_skills = roles_data.loc[roles_data['role'] == input_role, 'skills'].values[0]
    similarities = roles_data['skills'].apply(lambda skills: jaccard_similarity(input_skills, skills))
    recommended_roles = roles_data.loc[roles_data['role'] != input_role].copy()
    recommended_roles['similarity'] = similarities[roles_data['role'] != input_role]
    recommended_roles = recommended_roles.sort_values(by='similarity', ascending=False).head(top_n)
    return recommended_roles['role'].tolist()
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
        {'Java', 'Spring Boot', 'Microservices', 'System Design'},
        {'JavaScript', 'React', 'CSS', 'HTML'},
        {'JavaScript', 'Node.js', 'React', 'MongoDB'},
        {'C', 'C++', 'Microcontrollers', 'RTOS'},
        {'AWS', 'Azure', 'Cloud Security', 'Terraform'},
        {'SQL', 'ETL', 'Data Warehousing', 'Big Data'},
        {'Power BI', 'SQL', 'Dashboarding', 'Data Analysis'},
        {'Product Strategy', 'Agile', 'Stakeholder Management', 'Data Analytics'},
        {'CI/CD', 'Kubernetes', 'Cloud Computing', 'Infrastructure as Code'},
        {'Network Security', 'Penetration Testing', 'Risk Management', 'Cybersecurity Fundamentals'},
        {'Swift', 'Kotlin', 'Flutter', 'Mobile UI/UX'},
        {'Unity', 'C#', 'Game Physics', '3D Rendering'}
    ]
})
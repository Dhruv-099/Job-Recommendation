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
    'role': ['Data Scientist', 'ML Engineer', 'Data Analyst', 'AI Researcher', 'Software Engineer'],
    'skills': [
        {'Python', 'Machine Learning', 'Statistics'},
        {'Python', 'Deep Learning', 'TensorFlow'},
        {'SQL', 'Excel', 'Data Visualization'},
        {'Deep Learning', 'Neural Networks', 'Research'},
        {'Java', 'Software Development', 'Algorithms'}
    ]
})
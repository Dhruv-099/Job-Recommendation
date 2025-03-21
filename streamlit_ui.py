import streamlit as st
from recommendation_engine import recommend_roles, roles_data

st.title("Job Role Recommendation Engine")
selected_role = st.selectbox("Select a job role:", roles_data['role'])

if st.button("Get Recommendations"):
    recommended_roles = recommend_roles(selected_role, roles_data)
    if recommended_roles:
        st.write(f"Recommended roles for {selected_role}: {', '.join(recommended_roles)}")
    else:
        st.write("No similar roles found.")
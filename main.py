import streamlit as st
from PIL import Image
from css.style import apply_snorkel_style

# Customizing the page title and icon
st.set_page_config(
    page_title="Prompt Development Hub", page_icon=":rocket:", layout="wide"
)


# Apply the Snorkel style
st.markdown(apply_snorkel_style(), unsafe_allow_html=True)

# Snorkel logo
logo = Image.open("images/snorkel_logo.png")
st.image(logo, width=200)


# Main header
st.markdown(
    '<h1 class="main-header">Prompt Development Workflow</h1>', unsafe_allow_html=True
)

# Brief description
st.markdown(
    f"""
Welcome to the **Prompt Development Hub**. This platform is designed to streamline and enhance the process of prompt development, 
making it easier for you to create, refine, and iterate on prompts efficiently. 
Navigate to the section that best suits your current task:
"""
)

# Button section headers
st.markdown('<h2 class="section-header">Choose Your Task</h2>', unsafe_allow_html=True)

# Button links to other pages
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("Manual Annotation"):
    # Switch pages
    st.switch_page("pages/manual_annotations.py")
    st.rerun()

if st.button("Prompt Iteration"):
    st.switch_page("pages/prompt_iteration.py")
    st.rerun()

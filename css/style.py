import streamlit as st


def apply_snorkel_style():
    light_blue = "#84ceff"
    dark_blue = "#14194d"
    white = "#FFFFFF"

    snorkel_css = f"""
        <style>
            .header {{
                text-align: center;
                font-size: 2rem;
                color: {dark_blue};
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .sub-header {{
                font-size: 1.25rem;
                color: #555555;
                margin-top: 10px;
                margin-bottom: 10px;
            }}
            .response-container {{
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .stRadio>div {{
                display: flex;
                justify-content: space-evenly;
                margin-top: 10px;
                margin-bottom: 10px;
            }}
            .feedback-container {{
                margin-top: 10px;
                margin-bottom: 20px;
            }}
            .button-container {{
                display: flex;
                justify-content: center;
                margin-top: 30px;
            }}
            .stButton>button {{
                padding: 0.5rem 1.5rem;
                font-size: 1.2rem;
                background-color: {light_blue};
                color: {dark_blue};
                border: 2px solid {dark_blue};
                border-radius: 10px;
            }}
            .stButton>button:hover {{
                background-color: {dark_blue};
                color: {white};
            }}
            .footer {{
                text-align: center;
                font-size: 0.9rem;
                color: {dark_blue};
                margin-top: 50px;
            }}
        </style>
    """

    return snorkel_css

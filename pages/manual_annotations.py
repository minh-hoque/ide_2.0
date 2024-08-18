import streamlit as st
import pandas as pd
from css.style import apply_snorkel_style

# Page configuration
st.set_page_config(
    page_title="Manual Annotations", page_icon=":clipboard:", layout="wide"
)


# Apply the Snorkel style
st.markdown(apply_snorkel_style(), unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="header">Manual Annotations</h1>', unsafe_allow_html=True)

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Check if the required columns are present
    if set(["question", "response"]).issubset(df.columns):
        for index, row in df.iterrows():
            # Display context in an expander if it exists
            if "context" in df.columns:
                with st.expander(f"Context for Question {index+1}"):
                    st.write(row["context"])

            st.write(f"**Question {index+1}:** {row['question']}")
            st.write(f"**Response:** {row['response']}")

            # Rating section
            rating = st.radio(
                f"Rate the response for Question {index+1}:",
                ("ACCEPT", "REJECT"),
                key=f"rating_{index}",
            )

            if rating == "REJECT":
                # Ground truth response or feedback
                edited_gt = st.text_area(
                    "Provide the correct ground truth response (optional):",
                    key=f"ground_truth_{index}",
                )
                sme_feedback = st.text_area(
                    "Provide feedback for the data scientist:",
                    key=f"feedback_{index}",
                    placeholder="Enter your feedback here...",
                )

                # save the data into the dataframe
                df.loc[index, "rating"] = rating
                df.loc[index, "sme_feedback"] = sme_feedback
                df.loc[index, "edited_gt"] = edited_gt
            else:
                # if the rating is ACCEPT, then the edited_gt and sme_feedback are empty
                edited_gt = ""
                sme_feedback = ""
                df.loc[index, "rating"] = rating
                df.loc[index, "sme_feedback"] = sme_feedback
                df.loc[index, "edited_gt"] = edited_gt

            st.markdown("<hr>", unsafe_allow_html=True)

        # Final submission button
        if st.button("Submit Annotations"):
            st.success("Annotations submitted successfully!")
            # save the dataframe to a new csv file under /storage/manual_annotations
            df.to_csv(
                "./storage/manual_annotations/evaluated_responses.csv", index=False
            )
    else:
        st.error(
            "The uploaded CSV file must contain at least the 'question' and 'response' columns."
        )
else:
    st.info("Please upload a CSV file to begin annotating.")

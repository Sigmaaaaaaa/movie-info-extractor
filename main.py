import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- MODEL ----------------
model = ChatMistralAI(
    model="you-mistrial-chat-model"
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
You are an intelligent information extraction assistant.

Your task is to read the paragraph carefully and extract the most useful information about movies.

Extract the following details if they appear in the paragraph:

Movie Names:
Genres:
Themes:
Main Ideas:
Important Concepts:
Notable Cast (if mentioned):
Directors (if mentioned):
Years (if mentioned):
Keywords:

Then write a short summary (2-3 sentences) of the paragraph.
"""
    ),
    ("human",
    """
Instructions:
- Only extract information that is clearly mentioned or strongly implied.
- If some information is missing, write "Not mentioned".
- Keep the answer clean and well structured.

Paragraph:
{paragraph}

Extracted Information:
"""
    )
])

# ---------------- UI ----------------
st.title("Movie Information Extractor")

paragraph = st.text_area("Enter your paragraph")

if st.button("Extract Information"):

    if paragraph:
        final_prompt = prompt.invoke(
            {"paragraph": paragraph}
        )

        response = model.invoke(final_prompt)

        st.subheader("Extracted Information")
        st.write(response.content)

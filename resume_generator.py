import streamlit as st
import requests
import openai
from bs4 import BeautifulSoup

# Sidebar for API key input
st.sidebar.title("API Key Settings")
api_key_input = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key_input:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Set your OpenAI API key
client = openai.OpenAI(api_key=api_key_input)

# --- Step 1: Extract Job Links from HTML Content ---
st.title("AI Resume Generator from Job Description")

st.header("Step 1: Upload HTML Content Containing Job Links")
html_file = st.file_uploader("Upload HTML file", type=["html", "htm"])

job_links = []
job_map = {}

if html_file:
    soup = BeautifulSoup(html_file.read(), "html.parser")

    for a in soup.find_all("a", href=True):
        href = a['href']
        if "linkedin.com/jobs/view" in href:
            job_links.append(href)

    job_links = list(set(job_links))

    # Extract job titles and companies using AI
    job_display_list = []
    for link in job_links:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(link, headers=headers)
            page_text = BeautifulSoup(response.content, "html.parser").get_text(separator="\n", strip=True)

            ai_summary_prompt = f"""
You are an AI that extracts short summaries from LinkedIn job pages. Given this raw text, provide the job title and company name only, in the format:
Job Title: <title>\nCompany: <company>

Text:
{page_text[:3000]}
"""

            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You extract brief job summaries from job pages."},
                    {"role": "user", "content": ai_summary_prompt}
                ]
            )

            summary_lines = summary_response.choices[0].message.content.splitlines()
            title = next((line.split(":", 1)[1].strip() for line in summary_lines if line.lower().startswith("job title:")), "Unknown")
            company = next((line.split(":", 1)[1].strip() for line in summary_lines if line.lower().startswith("company:")), "Unknown")
            display = f"{title} at {company}"
            job_display_list.append(display)
            job_map[display] = link

        except Exception:
            continue

    if job_display_list:
        selected_display = st.selectbox("Step 2: Select a Job to Generate Resume", job_display_list)
        selected_link = job_map.get(selected_display)

        with st.form("candidate_form"):
            st.subheader("Step 3: Enter Your Profile Details")
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            location = st.text_input("Location")
            summary = st.text_area("Professional Summary")
            skills = st.text_area("Key Skills (comma-separated)")
            experience = st.text_area("Work Experience")
            education = st.text_area("Education")
            submitted = st.form_submit_button("Generate Resume")

        if submitted:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(selected_link, headers=headers)
                page_text = BeautifulSoup(response.content, "html.parser").get_text(separator="\n", strip=True)

                ai_extraction_prompt = f"""
You are an AI assistant that extracts structured information from job posting content. 
Given the following raw text from a LinkedIn job page, extract:
1. Job Title
2. Company Name
3. Full Job Description

Format your response as:
Job Title: <title>
Company: <company>
Job Description: <description>

Text:
{page_text[:8000]}
"""

                with st.spinner("Extracting job details with AI..."):
                    extraction_response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that extracts structured job data."},
                            {"role": "user", "content": ai_extraction_prompt}
                        ]
                    )

                    extracted = extraction_response.choices[0].message.content

                    title = "Unknown"
                    company = "Unknown"
                    job_description = ""

                    for line in extracted.splitlines():
                        if line.lower().startswith("job title:"):
                            title = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("company:"):
                            company = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("job description:"):
                            job_description = line.split(":", 1)[1].strip()
                            break

                    job_description += "\n" + "\n".join(extracted.splitlines()[extracted.splitlines().index(line)+1:])

                    st.subheader(f"Job Title: {title}")
                    st.subheader(f"Company: {company}")
                    st.text_area("Job Description", job_description[:3000], height=300)

                    st.subheader("Generated AI Resume")

                    prompt = f"""
You are a professional resume writer. Create a tailored resume in markdown format using the candidate's profile and the job description below.

Candidate Profile:
Name: {name}
Email: {email}
Phone: {phone}
Location: {location}
Summary: {summary}
Skills: {skills}
Experience: {experience}
Education: {education}

Job Title: {title}
Company: {company}

Job Description:
{job_description}

Make the resume focused, polished, and aligned with the job role. Format it cleanly with headers.
"""

                    with st.spinner("Generating resume with AI..."):
                        chat_response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that writes resumes."},
                                {"role": "user", "content": prompt}
                            ]
                        )

                        ai_resume = chat_response.choices[0].message.content
                        st.markdown(ai_resume)

            except Exception as e:
                st.error(f"Failed to extract job description or generate resume: {str(e)}")
    else:
        st.warning("No valid LinkedIn job links found in the uploaded HTML.")

st.markdown("""
<hr style="margin-top: 3em;">
<div style='text-align: center; font-size: 0.9em; color: grey;'>
    Built with ❤️ by <a href='https://www.linkedin.com/in/vaibhav-gupta-897096140/' target='_blank'>Vaibhav Gupta</a>
</div>
""", unsafe_allow_html=True)

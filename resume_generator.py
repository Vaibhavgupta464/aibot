import streamlit as st
import requests
import openai
from bs4 import BeautifulSoup
from io import BytesIO
from fpdf import FPDF
import google.generativeai as genai
import hashlib
import json
import time
from datetime import datetime
import re

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="AI Resume & Cold Email Generator")

# Initialize session state
if 'job_cache' not in st.session_state:
    st.session_state.job_cache = {}
if 'resume_history' not in st.session_state:
    st.session_state.resume_history = []
if 'email_history' not in st.session_state:
    st.session_state.email_history = []

# Sidebar for API key input and settings
st.sidebar.title("⚙️ Settings")
api_key_input = st.sidebar.text_input("Enter your OpenAI or Gemini API Key", type="password")

# Advanced settings
st.sidebar.subheader("Advanced Options")
use_cache = st.sidebar.checkbox("Use AI Response Caching", value=True, help="Cache AI responses to speed up re-processing")
max_jobs_to_process = st.sidebar.slider("Max jobs to process", 1, 20, 10, help="Limit number of jobs to process from HTML")
ai_model_choice = st.sidebar.selectbox("AI Model (if OpenAI)", ["gpt-4", "gpt-3.5-turbo"], help="Choose AI model for processing")

if not api_key_input:
    st.warning("⚠️ Please enter your API Key in the sidebar to continue.")
    st.stop()

# Helper functions
@st.cache_data
def generate_cache_key(text):
    """Generate a hash key for caching AI responses"""
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_response(cache_key, cache_type="job_summary"):
    """Get cached AI response"""
    if not use_cache:
        return None
    cache_key_full = f"{cache_type}_{cache_key}"
    return st.session_state.job_cache.get(cache_key_full)

def set_cached_response(cache_key, response, cache_type="job_summary"):
    """Store AI response in cache"""
    if use_cache:
        cache_key_full = f"{cache_type}_{cache_key}"
        st.session_state.job_cache[cache_key_full] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }

def call_ai_api(prompt, model_type="summary"):
    """Unified AI API calling function with caching"""
    cache_key = generate_cache_key(prompt)
    cached = get_cached_response(cache_key, model_type)
    
    if cached:
        st.sidebar.success(f"✅ Using cached {model_type} response")
        return cached['response']
    
    try:
        if api_key_input.startswith("sk-"):
            openai.api_key = api_key_input
            response = openai.ChatCompletion.create(
                model=ai_model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
        else:
            genai.configure(api_key=api_key_input)
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(prompt)
            result = response.text
        
        set_cached_response(cache_key, result, model_type)
        return result
    except Exception as e:
        st.error(f"AI API Error: {str(e)}")
        return None

def extract_linkedin_jobs(html_content):
    """Extract LinkedIn job links from HTML"""
    soup = BeautifulSoup(html_content, "html.parser")
    job_links = []
    
    for a in soup.find_all("a", href=True):
        href = a['href']
        if "linkedin.com/jobs/view" in href:
            # Clean up the URL
            job_id = re.search(r'/jobs/view/(\d+)', href)
            if job_id:
                clean_url = f"https://www.linkedin.com/jobs/view/{job_id.group(1)}"
                job_links.append(clean_url)
    
    return list(set(job_links))[:max_jobs_to_process]

def extract_hr_info(job_text):
    """Extract HR/recruiter information from job description"""
    hr_prompt = f"""
Analyze this LinkedIn job posting and extract any HR/recruiter contact information or hiring manager details.

Look for:
1. HR contact name
2. Recruiter name
3. Hiring manager name
4. Contact email
5. Any person mentioned as contact for applications
6. Department head names
7. Team lead names

Job Text:
{job_text[:4000]}

Provide the information in this format:
HR Contact: [Name if found, otherwise "Not specified"]
Email: [Email if found, otherwise "Not found"]
Title: [Their title/role if mentioned, otherwise "HR/Recruiter"]
Additional Info: [Any other relevant contact details]

If no specific contact is found, suggest generic titles like "Hiring Manager" or "HR Team".
"""
    
    return call_ai_api(hr_prompt, "hr_extraction")

def process_job_link(link, index):
    """Process a single job link and return job info"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(link, headers=headers, timeout=15)
        response.raise_for_status()
        
        page_text = BeautifulSoup(response.content, "html.parser").get_text(separator="\n", strip=True)
        
        ai_summary_prompt = f"""
Extract job information from this LinkedIn job page text. Provide ONLY:
Job Title: <exact title>
Company: <company name>
Location: <location if available>
Employment Type: <full-time/part-time/contract if mentioned>

Text (first 3000 characters):
{page_text[:3000]}
"""
        
        ai_response = call_ai_api(ai_summary_prompt, "job_summary")
        if not ai_response:
            return None
            
        # Parse AI response
        lines = ai_response.splitlines()
        title = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("job title:")), f"Job {index+1}")
        company = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("company:")), "Unknown Company")
        location = next((line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("location:")), "")
        
        # Create display name
        display_parts = [title, company]
        if location:
            display_parts.append(location)
        
        display_name = " | ".join(display_parts) + f" (#{index+1})"
        
        return {
            'display_name': display_name,
            'link': link,
            'title': title,
            'company': company,
            'location': location,
            'raw_text': page_text[:8000]  # Store for resume generation
        }
        
    except Exception as e:
        st.warning(f"⚠️ Could not process job {index+1}: {str(e)}")
        return {
            'display_name': f"Job {index+1} - Failed to Load Details",
            'link': link,
            'title': f"Job {index+1}",
            'company': "Unknown",
            'location': "",
            'raw_text': ""
        }

# --- Main Application ---
st.title("🚀 AI Resume & Cold Email Generator")
st.markdown("Generate tailored resumes and cold emails from LinkedIn job descriptions powered by AI")

# Display cache statistics
if use_cache and st.session_state.job_cache:
    st.sidebar.info(f"📊 Cache: {len(st.session_state.job_cache)} items stored")
    if st.sidebar.button("🗑️ Clear Cache"):
        st.session_state.job_cache.clear()
        st.success("Cache cleared!")

# Source selection
source_option = st.radio("📥 Choose data source:", ["Upload HTML", "Enter Job Link", "History"])

selected_job_info = None

if source_option == "Upload HTML":
    st.header("📄 Upload HTML with LinkedIn Job Links")
    html_file = st.file_uploader("Upload HTML file containing LinkedIn job links", type=["html", "htm"])

    if html_file:
        html_content = html_file.read()
        job_links = extract_linkedin_jobs(html_content)
        
        if not job_links:
            st.warning("No LinkedIn job links found in the uploaded HTML.")
        else:
            st.info(f"Found {len(job_links)} job links. Processing up to {max_jobs_to_process} jobs...")
            
            # Generate cache key for this HTML file
            html_hash = generate_cache_key(str(html_content))
            
            # Check if we've processed this HTML before
            if f"html_jobs_{html_hash}" in st.session_state.job_cache and use_cache:
                st.success("✅ Using cached job data for this HTML file")
                job_data_list = st.session_state.job_cache[f"html_jobs_{html_hash}"]['response']
            else:
                # Process jobs
                with st.spinner(f"🔄 Processing {len(job_links)} jobs..."):
                    progress_bar = st.progress(0)
                    job_data_list = []
                    
                    for i, link in enumerate(job_links):
                        progress_bar.progress((i + 1) / len(job_links))
                        st.text(f"Processing job {i+1}/{len(job_links)}...")
                        
                        job_info = process_job_link(link, i)
                        if job_info:
                            job_data_list.append(job_info)
                        
                        time.sleep(0.5)  # Rate limiting
                    
                    progress_bar.empty()
                    
                    # Cache the processed jobs
                    if use_cache:
                        st.session_state.job_cache[f"html_jobs_{html_hash}"] = {
                            'response': job_data_list,
                            'timestamp': datetime.now().isoformat()
                        }
            
            if job_data_list:
                job_options = [job['display_name'] for job in job_data_list]
                selected_display = st.selectbox("🎯 Select a job to work with:", job_options)
                
                if selected_display:
                    selected_job_info = next(job for job in job_data_list if job['display_name'] == selected_display)
                    
                    # Show job preview
                    with st.expander("👀 Job Preview", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Title:** {selected_job_info['title']}")
                            st.write(f"**Company:** {selected_job_info['company']}")
                        with col2:
                            st.write(f"**Location:** {selected_job_info['location'] or 'Not specified'}")
                            st.write(f"**Link:** {selected_job_info['link']}")

elif source_option == "Enter Job Link":
    st.header("🔗 Enter LinkedIn Job Link")
    job_url = st.text_input("Paste LinkedIn job URL here:")
    
    if job_url and "linkedin.com/jobs/view" in job_url:
        job_info = process_job_link(job_url, 0)
        if job_info:
            selected_job_info = job_info
            st.success(f"✅ Job loaded: {job_info['title']} at {job_info['company']}")

elif source_option == "History":
    st.header("📚 Generation History")
    
    # Create tabs for different history types
    history_tab1, history_tab2 = st.tabs(["📄 Resume History", "📧 Email History"])
    
    with history_tab1:
        if st.session_state.resume_history:
            history_options = [f"{item['job_title']} at {item['company']} - {item['timestamp']}" for item in st.session_state.resume_history]
            selected_history = st.selectbox("Select from previous resumes:", history_options)
            
            if selected_history:
                selected_item = st.session_state.resume_history[history_options.index(selected_history)]
                st.markdown("### Previous Resume")
                st.markdown(selected_item['resume'])
                
                if st.download_button("📥 Download Previous Resume", 
                                    selected_item['resume'], 
                                    file_name=f"resume_{selected_item['candidate_name']}.md",
                                    mime="text/markdown"):
                    st.success("Resume downloaded!")
        else:
            st.info("No resume history found. Generate some resumes first!")
    
    with history_tab2:
        if st.session_state.email_history:
            email_options = [f"{item['job_title']} at {item['company']} - {item['timestamp']}" for item in st.session_state.email_history]
            selected_email = st.selectbox("Select from previous emails:", email_options)
            
            if selected_email:
                selected_email_item = st.session_state.email_history[email_options.index(selected_email)]
                st.markdown("### Previous Cold Email")
                st.markdown(selected_email_item['email'])
                
                if st.download_button("📥 Download Previous Email", 
                                    selected_email_item['email'], 
                                    file_name=f"cold_email_{selected_email_item['sender_name']}.txt",
                                    mime="text/plain"):
                    st.success("Email downloaded!")
        else:
            st.info("No email history found. Generate some cold emails first!")

# Action selection - NEW FEATURE
if selected_job_info:
    st.header("🎯 Choose Your Action")
    action_choice = st.radio(
        "What would you like to generate?",
        ["📄 Generate Tailored Resume", "📧 Create Cold Email to HR"],
        help="Choose whether to create a resume or a cold email for this job opportunity"
    )
    
    # Common profile information form
    st.header("👤 Your Profile Information")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", help="Your full name")
            email = st.text_input("Email Address*", help="Your email address")
            phone = st.text_input("Phone Number", help="Contact phone number")
            location = st.text_input("Location", help="City, State/Country")
            
        with col2:
            linkedin_profile = st.text_input("LinkedIn Profile", help="Your LinkedIn URL (optional)")
            current_role = st.text_input("Current Role", help="Your current job title")
            experience_years = st.text_input("Years of Experience", help="Total years of relevant experience")
            
        if action_choice == "📄 Generate Tailored Resume":
            # Resume-specific fields
            st.subheader("📄 Resume Details")
            summary = st.text_area("Professional Summary", 
                                 help="Brief professional summary highlighting your key strengths",
                                 height=100)
            skills = st.text_area("Technical Skills", 
                                help="List your relevant technical skills (comma-separated)",
                                height=80)
            experience = st.text_area("Work Experience", 
                                    help="Describe your work experience, including job titles, companies, dates, and key achievements",
                                    height=150)
            education = st.text_area("Education", 
                                    help="Your educational background including degrees, institutions, and dates",
                                    height=100)
            
            # Resume customization options
            st.subheader("🎨 Resume Customization")
            col3, col4 = st.columns(2)
            
            with col3:
                resume_style = st.selectbox("Resume Style", 
                                          ["Professional", "Modern", "Creative", "ATS-Optimized"])
                include_keywords = st.checkbox("Optimize for ATS Keywords", value=True,
                                             help="Include relevant keywords from job description")
            
            with col4:
                resume_length = st.selectbox("Target Length", ["1 page", "2 pages", "No preference"])
                focus_areas = st.multiselect("Focus Areas", 
                                           ["Technical Skills", "Leadership", "Project Management", "Communication", "Problem Solving"],
                                           help="Areas to emphasize in the resume")
                
        else:  # Cold Email option
            # Email-specific fields
            st.subheader("📧 Cold Email Details")
            key_achievements = st.text_area("Key Achievements", 
                                          help="Highlight 2-3 major achievements relevant to this role",
                                          height=100)
            motivation = st.text_area("Why This Company/Role?", 
                                    help="Explain why you're interested in this specific company and role",
                                    height=80)
            value_proposition = st.text_area("What Value Can You Add?", 
                                           help="Describe what unique value you can bring to their team",
                                           height=80)
            
            # Email customization
            st.subheader("✉️ Email Customization")
            col5, col6 = st.columns(2)
            
            with col5:
                email_tone = st.selectbox("Email Tone", 
                                        ["Professional", "Friendly Professional", "Confident", "Humble"])
                email_length = st.selectbox("Email Length", 
                                          ["Concise (100-150 words)", "Medium (150-250 words)", "Detailed (250-350 words)"])
            
            with col6:
                include_attachments = st.checkbox("Mention Resume Attachment", value=True)
                request_meeting = st.checkbox("Request Informational Meeting", value=True)
        
        submitted = st.form_submit_button(f"🚀 Generate {action_choice.split(' ')[1]} {action_choice.split(' ')[2]}", use_container_width=True)

    if submitted:
        # Validation
        required_fields = [name, email]
        if not all(required_fields):
            st.error("❌ Please fill in all required fields (marked with *)")
        else:
            if action_choice == "📄 Generate Tailored Resume":
                # Generate resume (existing functionality)
                with st.spinner("🤖 Generating your tailored resume..."):
                    
                    resume_prompt = f"""
You are an expert resume writer and career coach. Create a highly tailored, professional resume based on the job requirements and candidate profile.

JOB INFORMATION:
- Title: {selected_job_info['title']}
- Company: {selected_job_info['company']}
- Location: {selected_job_info['location']}

CANDIDATE PROFILE:
- Name: {name}
- Email: {email}
- Phone: {phone}
- Location: {location}
- LinkedIn: {linkedin_profile}
- Current Role: {current_role}
- Experience: {experience_years} years
- Summary: {summary}
- Skills: {skills}
- Experience: {experience}
- Education: {education}

CUSTOMIZATION REQUIREMENTS:
- Style: {resume_style}
- Length: {resume_length}
- Focus Areas: {', '.join(focus_areas) if focus_areas else 'General'}
- ATS Optimization: {include_keywords}

JOB DESCRIPTION CONTEXT:
{selected_job_info['raw_text'][:6000]}

INSTRUCTIONS:
1. Create a compelling, well-structured resume in markdown format
2. Tailor content specifically to match the job requirements
3. {'Include relevant keywords naturally throughout the resume' if include_keywords else 'Focus on clear, professional language'}
4. Emphasize {', '.join(focus_areas) if focus_areas else 'relevant skills and experience'}
5. Use strong action verbs and quantify achievements where possible
6. Format should be clean and {resume_style.lower()}
7. Target length: {resume_length}

Provide a complete, ready-to-use resume that will help this candidate stand out for this specific role.
"""
                    
                    ai_resume = call_ai_api(resume_prompt, "resume_generation")
                    
                    if ai_resume:
                        # Display resume
                        st.success("✅ Resume generated successfully!")
                        st.markdown("### 📄 Your Tailored Resume")
                        st.markdown(ai_resume)
                        
                        # Save to history
                        resume_entry = {
                            'candidate_name': name,
                            'job_title': selected_job_info['title'],
                            'company': selected_job_info['company'],
                            'resume': ai_resume,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
                        st.session_state.resume_history.append(resume_entry)
                        
                        # Download options
                        col5, col6, col7 = st.columns(3)
                        
                        with col5:
                            st.download_button(
                                "📄 Download as Markdown",
                                ai_resume,
                                file_name=f"resume_{name.replace(' ', '_').lower()}.md",
                                mime="text/markdown"
                            )
                        
                        with col7:
                            # Generate clean text version
                            clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', ai_resume)
                            clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)
                            clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)
                            
                            st.download_button(
                                "📝 Download as Text",
                                clean_text,
                                file_name=f"resume_{name.replace(' ', '_').lower()}.txt",
                                mime="text/plain"
                            )
                        
                        # Resume analysis
                        with st.expander("📊 Resume Analysis", expanded=False):
                            word_count = len(ai_resume.split())
                            char_count = len(ai_resume)
                            
                            analysis_col1, analysis_col2 = st.columns(2)
                            with analysis_col1:
                                st.metric("Word Count", word_count)
                                st.metric("Character Count", char_count)
                            
                            with analysis_col2:
                                # Simple keyword matching
                                job_keywords = set(selected_job_info['raw_text'].lower().split())
                                resume_keywords = set(ai_resume.lower().split())
                                common_keywords = len(job_keywords.intersection(resume_keywords))
                                
                                st.metric("Keyword Matches", common_keywords)
                                st.metric("Estimated Read Time", f"{word_count // 200 + 1} min")

            else:  # Generate Cold Email
                with st.spinner("🤖 Generating your cold email..."):
                    
                    # First, extract HR information
                    hr_info = extract_hr_info(selected_job_info['raw_text'])
                    
                    # Generate cold email
                    email_prompt = f"""
You are an expert career coach and cold email specialist. Create a compelling, personalized cold email for a job opportunity.

JOB INFORMATION:
- Title: {selected_job_info['title']}
- Company: {selected_job_info['company']}
- Location: {selected_job_info['location']}

CANDIDATE PROFILE:
- Name: {name}
- Email: {email}
- Phone: {phone}
- Location: {location}
- LinkedIn: {linkedin_profile}
- Current Role: {current_role}
- Experience: {experience_years} years
- Key Achievements: {key_achievements}
- Motivation: {motivation}
- Value Proposition: {value_proposition}

EMAIL REQUIREMENTS:
- Tone: {email_tone}
- Length: {email_length}
- Include Resume Mention: {include_attachments}
- Request Meeting: {request_meeting}

HR/CONTACT INFORMATION FOUND:
{hr_info}

JOB DESCRIPTION CONTEXT:
{selected_job_info['raw_text'][:4000]}

INSTRUCTIONS:
1. Create a compelling cold email that stands out
2. Address it to the appropriate person (use HR info or generic "Hiring Manager")
3. Make it personal and specific to this company and role
4. Highlight relevant achievements and value proposition
5. Use a {email_tone.lower()} tone throughout
6. Keep it {email_length.lower()} length
7. Include a clear call-to-action
8. {'Mention attached resume' if include_attachments else 'Do not mention attachments'}
9. {'Request an informational meeting or call' if request_meeting else 'Focus on expressing interest'}
10. Make the subject line compelling and specific

Format:
Subject: [Compelling subject line]

[Email body]

Best regards,
{name}
{email}
{phone if phone else ''}
{linkedin_profile if linkedin_profile else ''}
"""
                    
                    ai_email = call_ai_api(email_prompt, "email_generation")
                    
                    if ai_email:
                        # Display email
                        st.success("✅ Cold email generated successfully!")
                        st.markdown("### 📧 Your Cold Email")
                        
                        # Display HR info if found
                        if hr_info and "not specified" not in hr_info.lower():
                            with st.expander("👥 HR Contact Information Found", expanded=True):
                                st.markdown(hr_info)
                        
                        st.markdown("### Email Content:")
                        st.markdown(ai_email)
                        
                        # Save to email history
                        email_entry = {
                            'sender_name': name,
                            'job_title': selected_job_info['title'],
                            'company': selected_job_info['company'],
                            'email': ai_email,
                            'hr_info': hr_info,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
                        st.session_state.email_history.append(email_entry)
                        
                        # Download and copy options
                        col8, col9, col10 = st.columns(3)
                        
                        with col8:
                            st.download_button(
                                "📧 Download Email",
                                ai_email,
                                file_name=f"cold_email_{selected_job_info['company'].replace(' ', '_').lower()}.txt",
                                mime="text/plain"
                            )
                        
                        with col9:
                            # Extract subject line for easy copying
                            subject_match = re.search(r'Subject:\s*(.+)', ai_email)
                            if subject_match:
                                subject_line = subject_match.group(1).strip()
                                st.text_input("📝 Subject Line (copy this):", value=subject_line)
                        
                        # Email analysis
                        with st.expander("📊 Email Analysis", expanded=False):
                            email_word_count = len(ai_email.split())
                            email_char_count = len(ai_email)
                            
                            email_analysis_col1, email_analysis_col2 = st.columns(2)
                            with email_analysis_col1:
                                st.metric("Word Count", email_word_count)
                                st.metric("Character Count", email_char_count)
                            
                            with email_analysis_col2:
                                st.metric("Estimated Read Time", f"{email_word_count // 200 + 1} min")
                                
                                # Check if email mentions company name
                                company_mentions = ai_email.lower().count(selected_job_info['company'].lower())
                                st.metric("Company Name Mentions", company_mentions)
                        
                        # Tips for sending the email
                        with st.expander("💡 Tips for Sending Your Cold Email", expanded=False):
                            st.markdown("""
                            **Before Sending:**
                            - Double-check the recipient's name and email address
                            - Review the email for any typos or errors
                            - Ensure your LinkedIn profile is up-to-date
                            - Prepare your resume in PDF format
                            
                            **Best Practices:**
                            - Send during business hours (9 AM - 5 PM)
                            - Tuesday to Thursday are often best days
                            - Follow up after 1 week if no response
                            - Keep follow-ups brief and add value
                            
                            **After Sending:**
                            - Connect with the person on LinkedIn
                            - Set a reminder to follow up
                            - Track your outreach in a spreadsheet
                            """)

# Statistics and footer
st.markdown("---")
col11, col12, col13, col14 = st.columns(4)

with col11:
    st.metric("Resumes Generated", len(st.session_state.resume_history))

with col12:
    st.metric("Emails Generated", len(st.session_state.email_history))

with col13:
    st.metric("Jobs Cached", len([k for k in st.session_state.job_cache.keys() if k.startswith('job_summary_')]))

with col14:
    if st.button("🔄 Reset All Data"):
        st.session_state.job_cache.clear()
        st.session_state.resume_history.clear()
        st.session_state.email_history.clear()
        st.success("All data reset!")

st.markdown("""
<div style='text-align: center; padding: 20px; color: grey;'>
    <p>🚀 <strong>AI Resume & Cold Email Generator</strong> - Powered by OpenAI & Gemini</p>
    <p>Built with ❤️ by <a href='https://www.linkedin.com/in/vaibhav-gupta-897096140/' target='_blank'>Vaibhav Gupta</a></p>
    <p><em>Generate tailored resumes and cold emails that get you noticed!</em></p>
</div>
""", unsafe_allow_html=True)

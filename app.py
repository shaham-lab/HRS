import os
from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client (will be None if no API key is set)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key) if api_key and api_key != 'your_openai_api_key_here' else None


@app.route('/')
def index():
    """Display the input form."""
    return render_template('index.html')


def get_health_recommendation(symptoms, duration, severity, additional_info):
    """Get health recommendations from OpenAI."""
    # Check if OpenAI client is initialized
    if client is None:
        # Return a demo response if no API key is configured
        return f"""**DEMO MODE - No OpenAI API key configured**

Based on your symptoms: {symptoms}

**Assessment:**
This is a demonstration response. To get real AI-powered health recommendations, please set up your OpenAI API key in the .env file.

**General Recommendations:**
1. Monitor your symptoms and note any changes
2. Stay hydrated and get adequate rest
3. Maintain a healthy diet
4. Take over-the-counter medications as appropriate

**When to Seek Medical Attention:**
- If symptoms persist for more than a few days
- If symptoms worsen or become severe
- If you experience any concerning or unusual symptoms

**Important:** This demo response is not actual medical advice. For real health recommendations, configure your OpenAI API key. Always consult with a qualified healthcare professional for proper diagnosis and treatment.
"""
    
    try:
        # Construct the prompt
        prompt = f"""You are a helpful medical assistant. Based on the following patient information, provide general health recommendations and suggestions. Remember to always advise consulting a healthcare professional.

Symptoms: {symptoms}
Duration: {duration}
Severity: {severity}/10
Additional Information: {additional_info if additional_info else 'None provided'}

Please provide:
1. A brief assessment of the symptoms
2. Possible causes (general information only)
3. Self-care recommendations
4. When to seek immediate medical attention
5. General lifestyle advice

Keep the response clear, concise, and easy to understand."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant who provides general health information and recommendations. Always remind users to consult healthcare professionals for proper diagnosis and treatment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting recommendation: {str(e)}. Please make sure your OPENAI_API_KEY is set correctly in the .env file."


@app.route('/output', methods=['POST'])
def output():
    """Display the output with AI-generated health recommendations."""
    if request.content_type and 'application/x-www-form-urlencoded' not in request.content_type and 'multipart/form-data' not in request.content_type:
        return "Invalid content type", 400
    
    # Get form data
    symptoms = request.form.get('symptoms', '').strip()
    duration = request.form.get('duration', '').strip()
    severity = request.form.get('severity', '').strip()
    additional_info = request.form.get('additional_info', '').strip()
    
    # Validate required fields
    if not symptoms or not duration or not severity:
        return render_template('index.html')
    
    # Get AI recommendation
    recommendation = get_health_recommendation(symptoms, duration, severity, additional_info)
    
    return render_template('output.html', 
                         symptoms=symptoms,
                         duration=duration,
                         severity=severity,
                         additional_info=additional_info,
                         recommendation=recommendation)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

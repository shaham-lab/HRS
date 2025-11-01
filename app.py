from flask import Flask, render_template, request
from llm_service import get_health_recommendation

app = Flask(__name__)


@app.route('/')
def index():
    """Display the input form."""
    return render_template('index.html')





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
    
    # Validate severity is a number between 1-10
    try:
        severity_int = int(severity)
        if not (1 <= severity_int <= 10):
            return render_template('index.html')
    except ValueError:
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

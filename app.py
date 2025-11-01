from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    """Display the input form."""
    return render_template('index.html')


@app.route('/output', methods=['POST'])
def output():
    """Display the output with the submitted text."""
    if request.content_type and 'application/x-www-form-urlencoded' not in request.content_type and 'multipart/form-data' not in request.content_type:
        return "Invalid content type", 400
    
    text_input = request.form.get('text_input', '').strip()
    
    if not text_input:
        return render_template('index.html')
    
    return render_template('output.html', text=text_input)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

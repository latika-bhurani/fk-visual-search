from flask import Flask

# instance of the flask class
# use the __ name __ here if using a single module
app = Flask(__name__)

@app.route('/sample')
def running():
	return 'Flask is running'

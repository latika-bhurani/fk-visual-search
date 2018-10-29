from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)


## methods will specify what kinds of http request will the webserver handle
## 'POST' specifies that the client application will send data along with the request to the webserver


@app.route('/hello', methods=['POST'])
def hello():
	message = request.get_json(force=True)
	name = message['name']
	# console.log(name)
	response = {
			'greeting': 'Hello, ' + name + '!'
	}
	return jsonify(response)
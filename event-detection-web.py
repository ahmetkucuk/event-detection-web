from flask import Flask
from flask_restful import Resource, Api
from detection.detect_full_disk import FullDiskDetector
import urllib
from flask import send_file
import sys
import os

app = Flask(__name__)
api = Api(app)

base_image_url = "http://dmlab.cs.gsu.edu/dmlabapi/images/SDO/AIA/2k/?wave={}&starttime={}"
image_base = ""
full_disk_detector = None


@app.route('/')
def home_page():
	return "Error: Use event type and starttime for getting response"


@app.route('/<string:event_type>/<string:starttime>')
def detect_events(event_type, starttime):
	wavelength = "193"
	if event_type == "AR":
		wavelength = "171"
	image_url = base_image_url.format(wavelength, starttime)
	result_image_name = event_type + "_" + wavelength + "_" + starttime

	input_image_filename = image_base + result_image_name + ".jpg"
	output_image_filename = image_base + result_image_name + "_labeled.jpg"

	if os.path.exists(output_image_filename):
		return send_file(output_image_filename, mimetype='image/gif')

	urllib.urlretrieve(image_url, input_image_filename)
	full_disk_detector.label_image(input_image_filename, output_image_filename)
	return send_file(output_image_filename, mimetype='image/gif')

if __name__ == '__main__':
	app.config["model_dir"] = sys.argv[1]
	app.config["image_dir"] = sys.argv[2]

	image_base = app.config["image_dir"]
	full_disk_detector = FullDiskDetector(app.config["model_dir"])

	app.run(host='0.0.0.0')

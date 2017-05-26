from flask import Flask
from flask_restful import Resource, Api
from detection.detect_full_disk import FullDiskDetector
import urllib
from flask import send_file

app = Flask(__name__)
api = Api(app)

base_image_url = "http://dmlab.cs.gsu.edu/dmlabapi/images/SDO/AIA/2k/?wave={}&starttime={}"
model_ckp = "/Users/ahmetkucuk/Documents/log_test/"
full_disk_detector = FullDiskDetector(model_ckp)


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
	input_image_filename = "/Users/ahmetkucuk/Documents/" + result_image_name + ".jpg"
	output_image_filename = "/Users/ahmetkucuk/Documents/" + result_image_name + "_labeled.jpg"
	urllib.urlretrieve(image_url, input_image_filename)
	full_disk_detector.label_image(input_image_filename, output_image_filename)
	return send_file(output_image_filename, mimetype='image/gif')

if __name__ == '__main__':
	app.run()

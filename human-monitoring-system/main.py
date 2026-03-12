import argparse, imutils
from lib.centroidtracker import CentroidTracker
from lib.trackableobject import TrackableObject
from lib.mailer import Mailer
from lib import config, thread
from imutils.video import VideoStream
from imutils.video import FPS
from itertools import zip_longest
import time, schedule, csv
import time, dlib, cv2, datetime
import numpy as np

t0 = time.time()

def run():

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.25,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	ap.add_argument("-d", "--distance", type=float, default=12.0,
		help="total queue distance in meters (default: 12.0)")
	ap.add_argument("-sp", "--spacing", type=float, default=0.7,
		help="average person spacing in meters (default: 0.7)")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# if a video path was not supplied, grab a reference to the ip camera
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0 # People who exited the queue
	totalUp = 0 # People who entered the queue
	x = []
	empty=[]
	empty1=[]

	# Variables for wait time prediction
	queue_distance = args["distance"]
	person_spacing = args["spacing"]
	queue_length = int(queue_distance / person_spacing)
	
	entry_times = {}
	exit_times = {}
	service_times = []
	avg_service_time = 0
	predicted_wait_time = 0

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width = 700)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw two horizontal lines:
		# Top line (Entry line) and Bottom line (Counter / Exit line)
		entry_y = int(H * 0.3)
		exit_y = int(H * 0.7)

		cv2.line(frame, (0, entry_y), (W, entry_y), (0, 255, 0), 2)
		cv2.putText(frame, "-Queue Entry-", (10, entry_y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

		cv2.line(frame, (0, exit_y), (W, exit_y), (0, 0, 255), 2)
		cv2.putText(frame, "-Counter Exit-", (10, exit_y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# We'll use the entry line (top) to count entrances and the exit line (bottom) to count exits
				# Assume people move from top to bottom (entering at the top, exiting at the bottom) -> positive direction=down
				# Or bottom to top (entering at bottom, exiting at top) -> negative direction=up
				# For a standard CCTV queue, let's assume they walk DOWN the screen.
				# Top (entry_y) -> Bottom (exit_y)

				# Check Entry (Top Line, moving down)
				if direction > 0 and centroid[1] > entry_y and objectID not in entry_times:
					totalUp += 1 # Total entered
					entry_times[objectID] = time.time()
					empty.append(totalUp)
					to.counted = True

				# Check Exit (Bottom Line, moving down)
				elif direction > 0 and centroid[1] > exit_y and objectID in entry_times and objectID not in exit_times:
					totalDown += 1 # Total exited
					exit_times[objectID] = time.time()
					
					# Calculate service time for this person
					service_time = exit_times[objectID] - entry_times[objectID]
					service_times.append(service_time)
					
					# Update average service time
					if len(service_times) > 0:
						avg_service_time = sum(service_times) / len(service_times)
						predicted_wait_time = queue_length * avg_service_time

					empty1.append(totalDown)
						
				x = []
				# compute current queue length (current people in between lines)
				current_queue_len = len(entry_times.keys()) - len(exit_times.keys())
				x.append(max(0, current_queue_len))


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		info = [
		("Enter", totalUp),
		("Exit", totalDown),
		("Current Queue", max(0, len(entry_times.keys()) - len(exit_times.keys()))),
		("Status", status),
		]

		info2 = [
		("Avg Service Time", f"{avg_service_time:.2f} sec"),
		("Est Wait Time", f"{predicted_wait_time:.2f} sec"),
		("Max Queue Length", queue_length),
		]

                # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Initiate a simple log to save data at end of the day
		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)
				
		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

		if config.Timer:
			# Automatic timer to stop the live stream. Set to 8 hours (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	if config.Thread:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()


# https://pypi.org/project/schedule/
if config.Scheduler:
	# Runs every day (09:00 am). You can change it.
	schedule.every().day.at("09:00").do(run)

	while 1:
		schedule.run_pending()
else:
	run()
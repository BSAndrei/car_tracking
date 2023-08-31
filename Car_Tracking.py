#Import opencv
import cv2

#Define the path to the cascade classifier for car detection
cascade_src = 'cars.xml'

#specify the path to the video file to be processed
video = r'D:\Projects\vehicle_camera_ai\video1.mp4'

#Constants for measurement reference
KNOWN_WIDTH = 1.8  # Known width of an object (e.g., a car) in meters
FOCAL_LENGTH = 800  # Example focal length in pixels

#function to calculate the distance to an object
def calculate_distance(known_width, focal_length, object_width_in_pixels):
    return (known_width * focal_length) / object_width_in_pixels

#Function to detect cars in a video and estimate their distances
def detectCars(filename):
    rectangles = []
    
    #Load cascade classifier for car detection
    cascade = cv2.CascadeClassifier(cascade_src)

    #open the video
    vc = cv2.VideoCapture(filename)

    #Check if the video capture is opened successfully
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    #Start processing each frame of the video
    while rval:
        rval, frame = vc.read()
        frameHeight, frameWidth, fdepth = frame.shape

        # Resize the frame to a specific width and height
        frame = cv2.resize(frame, (1000, 750))

        # Convert the frame to grayscale for Haar cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Haar cascade detection to detect cars in the grayscale frame
        cars = cascade.detectMultiScale(gray, 1.3, 3)

        for (x, y, w, h) in cars:
            #Drawing rectangles around the detected cars
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Calculate the estimate distance to each car
            object_width_pixels = w
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, object_width_pixels)
            distance_str = f'Distance: {distance:.2f} meters'

            # Display the estimated distance
            cv2.putText(frame, distance_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #show the result frame with rectangles and distance information
        cv2.imshow("Result", frame)

        #Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(33) == ord('q'):
            break

    # Release the video capture
    vc.release()

# Call the detectCars function to start car detection on the specified video
detectCars(video)

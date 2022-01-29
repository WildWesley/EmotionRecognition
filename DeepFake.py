import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace


def analyze_face(cv2_read_image):
    """This function takes in an image scanned in cv2 and"""
    """outputs a DeepFace prediction of different pieces"""
    """of information about the person in the image"""
    prediction = DeepFace.analyze(cv2_read_image)
    return prediction


def draw_rectangle(cv2_read_image, face_cascade):
    """This function takes an jpg image and uses cv2, along with"""
    """an algorithm called 'haar' in order to detect the location"""
    """of a face. Then it uses cv2 to draw a green rectangle around"""
    """the face it detects"""

    gray = cv2.cvtColor(cv2_read_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(cv2_read_image, (x, y), (x + w, y + h), (0, 255, 0), 10)
    # plt.imshow(cv2.cvtColor(cv2_read_image, cv2.COLOR_BGR2RGB))
    # plt.show()


def write_emotion(cv2_read_image, prediction):
    """This function will take in the image that we've scanned"""
    """with cv2, along with the dictionary with all of the info"""
    """that DeepFake came up with concerning the image, and it"""
    """will write the predicted emotion by the image"""

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        cv2_read_image,
        prediction["dominant_emotion"],
        (50, 50),
        font,
        2,
        (0, 0, 255),
        2,
        cv2.LINE_4,
    )
    # plt.imshow(cv2.cvtColor(cv2_read_image, cv2.COLOR_BGR2RGB))
    # plt.show()


def main():
    img = cv2.imread("image3.jpg")
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    prediction = analyze_face(img)
    draw_rectangle(img, faceCascade)
    write_emotion(img, prediction)


def open_web_cam():
    """We know that the web cam on our device will either be at"""
    """index 0 or index 1, depending on how many cameras we have"""
    """so we're going to accoung for both scenarios"""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap


# We'll open our webcam
cap = open_web_cam()
# We'll define our face cascade
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Now we'll create our loop that reads from the webcam
while True:
    # Reads one image from the webcam
    ret, frame = cap.read()
    try:
        # Detects the emotion only of a person
        emotion = DeepFace.analyze(frame, actions=["emotion"])

        draw_rectangle(frame, faceCascade)

        write_emotion(frame, emotion)

        # We show the video
        cv2.imshow("Demo video", frame)

    except:
        print("No face detected")

    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

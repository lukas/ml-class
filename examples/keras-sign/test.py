import numpy as np
import cv2
import argparse
import os
import keras
import signdata

def preprocess(gray_hand):
    # Be sure to do the same preprocessing here that you did in your training!
    gray_hand = gray_hand.astype('float32')
    gray_hand /= 255.
    return gray_hand


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_hands(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


mode = "fixed"
fist_model = load_detection_model("aGest.xml")
palm_model = load_detection_model("palm.xml")

hand_target_size = (28, 28)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="path to model file")
args = parser.parse_args()

if not os.path.exists(args.model):
    parser.error("The file %s does not exist!" % args.model)

model = keras.models.load_model(args.model)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.flip( gray_image, 1 )
    hands = []
    if mode == "detect":
        fists = detect_hands(fist_model, gray_image)
        palms = detect_hands(palm_model, gray_image)
    
        for bbox in fists:
            hands.append(bbox)
        for bbox in palms:
            hands.append(bbox)
    else:
        hands = [[100,100,300,300]]
    
    for bbox in hands:
        x, y, width, height = bbox
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        gray_hand = gray_image[x1:x2, y1:y2]
        try:
            gray_hand = cv2.resize(gray_hand, (hand_target_size))
            gray_hand = preprocess(gray_hand)
        except:
            print("There was an exception!")
            continue

        print("Max: ", np.max(gray_hand), "Min: ", np.min(gray_hand))

        gray_hand = np.expand_dims(gray_hand, 0)

        color = (255,255,255)
        draw_bounding_box(bbox, gray_image, color)
        preds = model.predict(gray_hand)
        p = np.argmax(preds)
        prob = np.max(preds)
        text = signdata.letters[p] + " " + str(prob)
        text_box = (bbox[0], bbox[1]+bbox[3]+60)
        draw_text(text_box,gray_image,text,color)
        
    # Display the resulting frame
    cv2.imshow('frame',gray_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

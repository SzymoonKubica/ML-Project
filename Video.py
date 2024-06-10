import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')
model.load_weights('model_weights.h5')

label_mapping={0: 'zly',
 1: 'pogardliwy',
 2: 'zniesmaczony',
 3: 'przerazony',
 4: 'szczesliwy',
 5: 'neutralny',
 6: 'smutny',
 7: 'zaskoczony'}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()
    face_rec = face_cascade.detectMultiScale(face_img)

    if len(face_rec) == 0:
        return face_img, None, None, None
    
    for (x, y, w, h) in face_rec:
        cv2.rectangle(face_img,
                    (x, y), 
                    (x + w, y + h), 
                    color=(0, 0, 255), 
                    thickness=4)
        face = face_img[y:y + h, x:x + w]
        return face_img, face, x, y

    return face_img, None, None, None

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame,face, x, y = detect_face(frame)

    if x == None or y == None:
        cv2.imshow('Video face detect',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face_for_model = cv2.resize(face,(96,96))
        face_for_model = face_for_model / 255.0
        face_for_model = np.expand_dims(face_for_model, axis=0)  
        face_for_model = face_for_model.astype('float32')

        predicted_face = model.predict(face_for_model)
        right_face = np.argmax(predicted_face, axis = 1)
        right_face_label = right_face[0]

 
        cv2.putText(frame,
                    text=label_mapping[right_face_label],
                    org = (x-15,y-15),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX,
                    fontScale = 1/2,
                    color = (0,0,255),
                    thickness = 2 )
     
        cv2.imshow('Video face detect',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
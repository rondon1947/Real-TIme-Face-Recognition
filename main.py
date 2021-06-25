import cv2
import numpy as np
import os


def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda X: X[0])[:k]
    label = np.array(dk)[:, -1]
    output = np.unique(label, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip = 0
face_data = []
labels = []
dataset_path = './outputData/'
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("Loaded File: " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_labels), axis=1)


while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        predictedName = names[int(out)]
        cv2.putText(frame, predictedName, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)

    cv2.imshow("Faces", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

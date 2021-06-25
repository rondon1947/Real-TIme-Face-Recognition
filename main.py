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

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
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
        print("Loaded File: " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

# print(face_dataset.shape)
# print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
# print(trainset.shape)














#
# while True:
#     ret, frame = cap.read()
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if not ret:
#         continue
#
#     faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
#     faces = sorted(faces, key=lambda f: f[2] * f[3])
#     face_section = frame
#     for (x, y, w, h) in faces[-1:]:
#         cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)
#         offset = 10
#         face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
#         face_section = cv2.resize(face_section, (100, 100))
#         skip += 1
#         if skip % 10 == 0:
#             face_data.append(face_section)
#             print(len(face_data))
#     cv2.imshow("Video Frame", frame)
#     cv2.imshow("Face Section", face_section)
#     key_pressed = cv2.waitKey(1) & 0xFF
#     if key_pressed == ord('q'):
#         break
#
# face_data = np.asarray(face_data)
# face_data = face_data.reshape((face_data.shape[0], -1))
# print(face_data.shape)
# np.save(dataset_path + file_name + '.npy', face_data)
# print("Data Successfully saved")
# cap.release()
# cv2.destroyAllWindows()

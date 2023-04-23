import cv2
import numpy as np
import datetime
import smtplib
import imghdr
from email.message import EmailMessage
import face_recognition

# Загрузка классификатора Haar Cascade для распознавания инструмента
tool_cascade = cv2.CascadeClassifier('tool_cascade.xml')

# Функция для отправки отчета на почту
def send_email(image_path):
    sender_email = "sender@example.com"
    receiver_email = "receiver@example.com"
    password = "password"
    message = EmailMessage()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = 'Отчет о возврате инструмента'
    message.set_content('Отчет о возврате инструмента прилагается')
    with open(image_path, 'rb') as f:
        file_data = f.read()
        file_type = imghdr.what(f.name)
        file_name = f.name
    message.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, password)
        smtp.send_message(message)

# Функция для отправки уведомления о возврате инструмента на стенд
def send_notification():

    def send_email(to, subject, body):
        gmail_user = "your_email@gmail.com" # замени на свой email
        gmail_password = "your_password" # замени на свой пароль
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(gmail_user, gmail_password)
            message = 'Subject: {}\n\n{}'.format(subject, body)
            server.sendmail(gmail_user, to, message)
            server.quit()
            print("Отчет успешно отправлен!")
        except:
            print("Не удалось отправить отчет по почте.")

# Функция для распознавания лиц
def recognize_face(frame, encodings, names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        face_distances = face_recognition.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = names[best_match_index]
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

# Основной код
cap = cv2.VideoCapture(0)

# Инициализация лицевых кодировок и имен для распознавания лиц
encodings = []
names = []

# Загрузка фотографий и кодировок известных лиц
known_face_names = ['John', 'Jane']
for name in known_face_names:
    img = cv2.imread(f'{name}.jpg')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(rgb)[0]
    encodings.append(encoding)
    names.append(name)

    # Загрузка изображения с инструментом
    frame = cv2.imread('tool_image.jpg')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка изображения для определения наличия инструмента на стенде
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Обнаружение инструмента на изображении
    tools = tool_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Отображение области с инструментом на изображении
    for (x, y, w, h) in tools:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Отображение результата
cv2.imshow('Tool detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#получение текущей даты и времени для использования в именах файлов
now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

#цикл по всем контурам
for contour in contours:
# вычисление площади контура
    area = cv2.contourArea(contour)

MIN_AREA = 100  # присваиваем значение 100 переменной MIN_AREA
# игнорирование слишком маленьких контуров
if area < MIN_AREA: (x, y, w, h) = cv2.boundingRect(contour)

# выделение области с инструментом на изображении
instrument_roi = gray[y:y+h, x:x+w]

# распознавание лица в области с инструментом
face = recognize_face(instrument_roi)

# сохранение изображения с выделенной областью и, если было распознано лицо, с его обведенным прямоугольником
cv2.imwrite(f'ROI_{date_time}_{x}_{y}.jpg', instrument_roi)
if face is not None:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f'ROI_with_face_{date_time}_{x}_{y}.jpg', gray)

    # отправка уведомления о возврате инструмента на стенд на указанный email
    send_email('notification@example.com', 'Return of tool', f'Tool with ID {x}_{y} has been returned to the stand by {face}.')
else:
    # отправка уведомления о возврате инструмента на стенд без указания имени пользователя, если лицо не распознано
    send_email('notification@example.com', 'Return of tool', f'Tool with ID {x}_{y} has been returned to the stand.')
    #отображение и сохранение обработанного изображения
cv2.imshow("Processed Image", gray)
cv2.imwrite(f'Processed_Image_{date_time}.jpg', gray)

#ожидание нажатия клавиши для завершения работы
cv2.waitKey(0)
# освобождаем ресурсы
cv2.destroyAllWindows()
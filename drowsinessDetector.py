import cv2
import dlib
from scipy.spatial import distance
from pygame import mixer

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
mixer.init()
warningSound = mixer.Sound("./Scream.wav")

#status marking for current state
sleep = 0
# drowsy = 0
active = 0
status=""
color=(0,0,0)
warning = False

# def calculate_EAR(eye):
# 	A = distance.euclidean(eye[1], eye[5])
# 	B = distance.euclidean(eye[2], eye[4])
# 	C = distance.euclidean(eye[0], eye[3])
# 	EAR = (A+B)/(2.0*C)
# 	if(EAR>0.22):
# 		return 2
# 	elif(EAR>0.18 and EAR<=0.22):
# 		return 1
# 	else:
# 		return 0

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	EAR = (A+B)/(2.0*C)
	return EAR

def calculate_MAR(mount): # 49 55 52 58
	A = distance.euclidean(mount[0], mount[9])
	B = distance.euclidean(mount[6], mount[9])
	C = distance.euclidean(mount[3], mount[9])
	return ((A + B) / C)


while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = hog_face_detector(gray)
	for face in faces:

		face_landmarks = dlib_facelandmark(gray, face)
		leftEye = []
		rightEye = []
		mount = []

		for n in range(36,42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x,y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

		for n in range(42,48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x,y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
		
		for n in range(49, 61):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			cv2.circle(frame,(x,y),0,(0,255,0),-1)
			mount.append((x,y))
		
		leftEar = calculate_EAR(leftEye)
		rightEar = calculate_EAR(rightEye)
		earBlink = (leftEar + rightEar)/2

		MAR = calculate_MAR(mount)

		if(earBlink < 0.21 and MAR < 1.9):
			sleep+=1
			# drowsy=0
			active=0
			if(sleep>20):
				warning = True
				status="DIEEEE !!!"
				color = (255,0,0)
			elif (sleep>10):
				status="Blink detected !"
				color = (255,0,0)

		# elif(leftEar==1 or rightEar==1):
		# 	sleep=0
		# 	active=0
		# 	drowsy+=1
		# 	if(drowsy>6):
		# 		warning = False
		# 		status="You'll die !"
		# 		color = (0,0,255)

		else:
			# drowsy=0
			sleep=0
			active+=1
			if(active>6):
				warning = False
				status="You're ok"
				color = (0,255,0)

		
		warningSound.play(-1) if warning else warningSound.stop()
		cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)
		cv2.putText(frame, f"EAR: {round(earBlink,2)}", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)
		cv2.putText(frame, f"MAR: {round(MAR, 2)}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)

	cv2.imshow("Are you Sleepy", frame)

	key = cv2.waitKey(1)
	if key == 27: #Esc key
		break
cap.release()
cv2.destroyAllWindows()
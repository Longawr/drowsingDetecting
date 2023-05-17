import cv2
import dlib
from scipy.spatial import distance
from pygame import mixer
import tkinter as tk
from PIL import ImageTk, Image


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mixer.init()
warningSound = mixer.Sound("Scream.wav")

#khởi tạo root
root = tk.Tk()
#thông số mắt
EAR_threshold = 0.2
MAR_threshold = 1.8


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

# Hàm xử lý sự kiện khi nhấn nút "Lưu"
def save_parameters():
    global EAR_threshold, MAR_threshold
    EAR_threshold = float(EAR_entry.get())
    MAR_threshold = float(MAR_entry.get())
    adjust_window.destroy()

# Hàm hiển thị giao diện điều chỉnh tham số mắt
def show_adjust_window():
    global adjust_window, EAR_entry, MAR_entry
    adjust_window = tk.Toplevel()
    adjust_window.title("Điều chỉnh tham số mắt")
    adjust_window.resizable(False, False)
    adjust_window.geometry("200x100+810+130")

    # Tạo các phần tử trong giao diện
    ear_label = tk.Label(adjust_window, text="Ngưỡng EAR:")
    ear_label.pack()
    EAR_entry = tk.Entry(adjust_window)
    EAR_entry.insert(0, str(EAR_threshold))
    EAR_entry.pack()

    mar_label = tk.Label(adjust_window, text="Ngưỡng MAR:")
    mar_label.pack()
    MAR_entry = tk.Entry(adjust_window)
    MAR_entry.insert(0, str(MAR_threshold))
    MAR_entry.pack()

    save_button = tk.Button(adjust_window, text="Lưu", command=save_parameters)
    save_button.pack()


def Warning_Sleep():
	#status marking for current state
	sleepCount = 0
	# drowsy = 0
	wakeUpCount = 0
	status=""
	color=(0,0,0)
	warning = False
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
			
			leftEAR = calculate_EAR(leftEye)
			rightEAR = calculate_EAR(rightEye)
			avgEAR = (leftEAR + rightEAR)/2

			mar = calculate_MAR(mount)

			if(avgEAR < EAR_threshold and mar < MAR_threshold):
				sleepCount+=1
				# drowsy=0
				wakeUpCount=0
				if(sleepCount>30):
					warning = True
					status="Sleep !!!"
					color = (255,0,0)
				elif (sleepCount>10):
					status="Blink !"
					color = (255,0,0)

		
			elif(avgEAR > 0.22 or mar > 1.85):
				# drowsy=0
				sleepCount=0
				wakeUpCount+=1
				if(wakeUpCount>6):
					warning = False
					status="You're ok"
					color = (0,255,0)

			
			warningSound.play(-1) if warning else warningSound.stop()
			cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)
			cv2.putText(frame, f"EAR: {round(avgEAR,2)}", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)
			cv2.putText(frame, f"MAR: {round(mar, 2)}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)

		cv2.imshow("Are you Sleepy", frame)

		key = cv2.waitKey(1)
		if key == 27: #Esc key
			break
	cap.release()
	cv2.destroyAllWindows()

def quit():
    root.destroy()
    
def main():
	root.title("Are you Sleepy")
	root.geometry("500x400+400+150")
	root.resizable(False, False)
	root.configure(bg="pink")

	#Create background image
	image = Image.open("bg.jpg")
	bg = ImageTk.PhotoImage(image)
	
	# Create a Label widget with the image
	label = tk.Label(root,text="Chào mừng đến với ứng dụng của nhóm chúng tôi!", font=("Arial", 15,"bold"),bg="green", fg="white")
	label.pack()
	label1 = tk.Label(root,image=bg)
	label1.pack(pady=10)
	# create start button
	start_button = tk.Button(root, text="Start", font=("Arial", 16), bg="green", fg="white",
							command=Warning_Sleep)
	start_button.place(x=80,y=350)

	# create settings button
	settings_button = tk.Button(root, text="Settings", font=("Arial", 14), bg="blue", fg="white",
								command=show_adjust_window)
	settings_button.place(x= 230, y=350)

	# create exit button
	exit_button = tk.Button(root, text="Exit", font=("Arial", 14), bg="red", fg="white",
							command=quit)
	exit_button.place(x=400,y=350)

	root.mainloop()

#Main loop
main()
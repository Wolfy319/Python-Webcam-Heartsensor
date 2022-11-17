import time
import cv2
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
heart_values = []
heart_times = []
max_freqs = []
framerate = 30
fig = plt.figure()
ax = fig.add_subplot(111)
timer = 0
currtime = time.time()
interval = 10.0
maximums = []
numavg = 1
bpmText = "Calculating..."

def generate_frequencies(heart_values, samples):
    step = samples / interval
    start = int(0.83/step)
    end = int(3.4/step)
    freq = np.fft.rfft(heart_values, samples)
    timing = np.fft.fftfreq(freq.size, 1/step)

    index = np.where((timing > 0.82) & (timing < 3.5))
    timing = timing[index]
    freq = freq[index]
    absfreq = np.abs(freq)
    max = np.argmax(absfreq)
    maximums.append(timing[max])
    if len(maximums) > 100:
        maximums.pop(0)

    
    mag = np.sqrt(np.square(freq.real) + np.square(freq.imag))
    # if len(max_freqs)>= 50:
    #     max_freqs.pop(0)
    #     print(np.average(max_freqs))
    # max_freqs.append(max/50)
    return absfreq, timing, mag

def animate(x, y, heart_times, heart_values):
    heart_times.append(x)
    heart_values.append(y)
    samples = len(heart_times)
    if x - heart_times[0] > interval:
        heart_times.pop(0)
        heart_values.pop(0)
        freq,times,mag = generate_frequencies(heart_values, samples)        
        ax.plot(times, freq) 
        bpm = int(np.average(maximums) * 60)
        print(bpm)
        global bpmText 
        bpmText = str(bpm)
        ## Perform FFT with SciPy
    
    ax.set_autoscaley_on(True)
    ax.set_autoscalex_on(True)
    # ax.set_xlim((0.34,3.5))
    fig.canvas.draw()
    plot_img_np = np.frombuffer(fig.canvas.tostring_rgb(),
                                dtype=np.uint8)
    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()
    cv2.imshow('Graph', plot_img_np)
    # plt.clf()
    # plt.plot(heart_times, heart_values)
    # plt.pause(0.001)
# Initialize webcam video
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    raise IOError("can't open webcam")

heart_value = 0
while(True):
    _, frame = webcam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    roi = []
    roi_height = roi_width = 0
    bound_boxes = frame.copy()
    
    for x,y,w,h in faces:
        roi = frame[y+5:y+50, x+50:x+150]
        roi_height, roi_width, _ = roi.shape
        bound_boxes = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
        bound_boxes = cv2.rectangle(frame, (x+(w//3), y), (x+(w*2//3), y+(y//4)), (255,0,0), 3)
        heart_value = np.average(roi[:][:][1])
        # /(roi_height*roi_width)
    
    height,width,_ = frame.shape
    prevtime = currtime
    currtime = time.time()
    timer = timer + (currtime - prevtime)
    # print(timer)
    bpmText = animate(timer,heart_value,heart_times,heart_values)
    # height, width, _ = frame.shape
    # roi_box = cv2.rectangle(frame,(width//2 - width//12,height*3//12),(width//2 + width//12,height*4//12),(0,0,255),3)
    # print(height // 50, width // 100)
    bpmstr = "BPM - {bpm}".format(bpm = bpmText)
    cv2.putText(bound_boxes, bpmstr, (height - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, 2)
    cv2.imshow("faces", bound_boxes)
    # cv2.imshow("faces", roi)

    c = cv2.waitKey(1)
    if c == 27 :
        break
webcam.release()
cv2.destroyAllWindows

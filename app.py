from flask import Flask,render_template,request
import pickle
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
app= Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/StartTest',methods=['POST','GET'])
def StartTEST():
    if request.method=="POST":
        fn=request.form["Name"]
        mask_on='Mask On..'+'Welcome '+fn
        mask_off='No Mask..'+'Sorry '+fn
        print(mask_on)
        print(mask_off)
        with open('my_model','rb') as f:
            model = pickle.load(f)
        path1=r'C:\Users\ayush\Desktop\flask\data.xml'
        haar_data = cv2.CascadeClassifier(path1)
        capture = cv2.VideoCapture(0)  
        data = []
        font = cv2.FONT_HERSHEY_DUPLEX
        while True:
            flag,img = capture.read()
            if flag:
                faces = haar_data.detectMultiScale(img)
                for x,y,w,h in faces:
                    face = img[y:y+h,x:x+w,:]
                    face = cv2.resize(face,(50,50))
                    face = face.reshape(1,-1)
                    pred = model.predict(face)[0]
                    if pred==0:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
                        cv2.putText(img,'MASK',(x,y),font,1,(0,255,0),2)
                        text = mask_on
                        # get the width and height of the text box
                        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=1.5, thickness=1)[0]
                        # set the text start position
                        text_offset_x = 10
                        text_offset_y = img.shape[0] - 25
                        # make the coords of the box with a small padding of two pixels
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
                        cv2.rectangle(img, box_coords[0], box_coords[1], (255,255,255), cv2.FILLED)
                        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=1.5, color=(0, 0, 0), thickness=1)

                    else:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
                        cv2.putText(img,'NO MASK',(x,y),font,1,(0,0,255),2)
                        text = mask_off
                        # get the width and height of the text box
                        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=1.5, thickness=1)[0]
                        # set the text start position
                        text_offset_x = 10
                        text_offset_y = img.shape[0] - 25
                        # make the coords of the box with a small padding of two pixels
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
                        cv2.rectangle(img, box_coords[0], box_coords[1], (255,255,255), cv2.FILLED)
                        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=1.5, color=(0, 0, 0), thickness=1)
                cv2.imshow('result',img)
                if cv2.waitKey(2) == 27:break
        capture.release()
        cv2.destroyAllWindows()
        # print(fn,ln)
        # return "Submitted Successfully"
    return render_template('StartTest.html')

@app.route('/profile/<string:name>')
def profile(name):
    return 'Hello'+str(name)

@app.route('/profile/<int:id>')
def profile1(id):
    return 'Your id is '+str(id)

if __name__=="__main__":
    app.run(debug=True)
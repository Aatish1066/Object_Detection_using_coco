import cv2

thresh = 0.7
cap = cv2.VideoCapture(0)
cap.set(3 ,640)
cap.set(4 ,480)

classfile = 'coco.names'
with open(classfile ,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
configpath = 'op.prototext'
weightspath = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn_DetectionModel(weightspath ,configpath)
# noinspection PyArgumentList
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)
while True:
    sucess ,img = cap.read()
    classid ,conf ,bbox = net.detect(img ,confThreshold=thresh)
    if len(classid) != 0:
        for classid ,confidence ,box in zip(classid.flatten() ,conf.flatten() ,bbox):
            cv2.rectangle(img ,box ,color=(255.0 ,0) ,thickness=2,lineType=None,shift=None)
            cv2.putText(img ,classnames[classid - 1] ,(box[0] + 10 ,box[1] + 30) ,cv2.FONT_HERSHEY_PLAIN ,2 ,
                        (0 ,225 ,0) ,2)
    cv2.imshow("output" ,img)
    if cv2.waitKey(100) & 0xFF == ord(' '):
        cv2.imwrite('test.jpg' ,img)
        break
cap.release()
cv2.destroyAllWindows()

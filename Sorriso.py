import cv2
import sys

img = sys.argv[1]
cascPath = "haarcascade_smile.xml"

smile = cv2.CascadeClassifier(cascPath)

img = cv2.imread(img)
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smiles = smile.detectMultiScale(cinza, 5, 5)

for (x,y,w,h) in smiles:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

print("Foram encontrados " + str(len(smiles)) + " sorrisos")

cv2.imshow("Rostos encontrados",img)
cv2.waitKey(0)

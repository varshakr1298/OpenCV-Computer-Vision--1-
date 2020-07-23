import cv2
import matplotlib.pyplot as plt
flower=cv2.imread('sunflower.jpg')
#cv2.imshow('sunflower',flower)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(flower)
plt.show()
mask=cv2.imread('mask.png')
#cv2.imshow('mask',mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(mask)
plt.show()
# Performing Inpainting
ns = cv2.inpaint(flower, mask, 1, cv2.INPAINT_NS) 
telea = cv2.inpaint(flower, mask, 1, cv2.INPAINT_TELEA)

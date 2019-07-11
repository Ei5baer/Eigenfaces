
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import cv2
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

image_filenames = {}
for path,dirs,files in os.walk("data_small"):
    for file in files :
        if ( file.endswith('jpg') ) :
            #image_filenames.append( path+ "\\" +file )
            tokens = path.split("\\")
            key = tokens[-1]
            image_filenames[key]= path + "\\" +file 
print( image_filenames )
#exit()


images = {}
cropped_faces = []

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for name in image_filenames :
    images[name] = cv2.imread( image_filenames[name] )
    test_image = cv2.imread(image_filenames[name])
    
# Converting to grayscale as opencv expects detector takes in input gray scale images
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying grayscale image
    cv2.waitKey()
    #cv2.imshow( 'person',test_image_gray)
    




    faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

# Let us print the no. of faces found
    print('Faces found: ', len(faces_rects))

# Our next step is to loop over all the co-ordinates it returned and draw rectangles around them using Open CV.We will be drawing a green rectangle with thicknessof 1
# In[148]:


    #cropped_faces = []
    for (x,y,w,h) in faces_rects:
        a = x//1 + w//2 -50
        b = y//1 + h//2 - 50
        c = x//1 + w//2 + 50
        d = y//1 + h//2 + 50
        cv2.rectangle(test_image, (a, b),( c, d) , (1, 1, 1), 1)
        test_image = convertToRGB(test_image)
        crop_img = test_image[a:c, b:d]
        #cv2.imshow("cropped_face", crop_img)
        cropped_faces.append(crop_img)
       
    test_image = convertToRGB(test_image)
    #cv2.imshow("person", test_image)

equ = []

print (len( cropped_faces ))

face_vec_list = []

 
for face in cropped_faces :
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    
   
    dimensions = face_gray.shape
    
    
    
    face_eq = cv2.equalizeHist(face_gray)

    
    face_vec = face_eq.flatten()

    
    face_list_vec = list(face_vec)
    
    face_vec_list.append( face_list_vec)

  
    
fva = np.array(face_vec_list)
orig_fva = np.array( fva )
s = np.shape( fva)
print (s)
mean_fva = fva.mean(axis = 0)

fva = fva - mean_fva
fvat = fva.T
#t = np.shape(fvat)
#print(t)
Cov_mat = np.dot(fva, fvat)
a = np.shape(Cov_mat)
print (a)

eigval, eigvec=np.linalg.eig(Cov_mat)

print ("eigvals are " , eigval)
input( "proceed ? press any key " ) 

list_eigvals = []

for x in np.nditer(eigval):
    x_abs = abs(x)
    list_eigvals.append(x_abs)

eigvals = np.array(list_eigvals)

ind  = np.argsort(eigvals)

sort_eigvals = np.sort(eigvals)

eigveclist = eigvec.tolist()
print (eigvec)
a = ind[-1]
eigfac = np.array(eigveclist[a])

eigenfaces = []



for x in range (-2,-12,-1):
    a=ind[x]
    eigfac = np.vstack((eigfac,np.array(eigveclist[a])))

eigfac_correct = eigfac.T

#print (eigfac_correct)

eigenfac = np.dot(fvat,eigfac_correct)

for i in range( eigenfac.shape[1] )  :
    nrm =  np.linalg.norm ( eigenfac[:,i] )
    print ("norm is ", nrm )
    eigenfac[ : ,i ] = eigenfac[:,i] /  nrm 


s = np.shape(eigenfac)
#print(s)
print ("Bark")
#work_eig = eigenfac.T

implist = []
print ("dog")

for g in range (0 , s[1]):
    f = eigenfac[:,g]
    print ( "l2-norm of  f is ", np.linalg.norm( f ) )
    
    m = f.reshape(100,100)
    implist.append(m)
    print ("Bear")

for f in face_vec_list :
    f_approx = np.dot ( np.array( f)-mean_fva , np.dot( eigenfac , eigenfac.T ) ) + mean_fva
   
    cv2.imshow( "image" , (np.array(f) ).reshape(100,100) )
    cv2.waitKey()
    
    cv2.imshow( "image" , ((np.array(f_approx ))  / 255.0   ).reshape(100,100) )
    cv2.waitKey()
    yesorno = input("proceed (y/n) ? ")
    if ( yesorno == 'n' ) :
        exit()

for i in range (s[1]):
    cv2.imshow("image", implist[i])
    cv2.waitKey()


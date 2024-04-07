from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.filters import gabor
from statistics import mean
import os
import cv2
import numpy as np
import csv

for file in os.listdir('CroppedImgs/'):
    k_value = ""     
    file_path = f'CroppedImgs/{file}'
    img = cv2.imread(file_path)
    base_name = file.split("_")[0]
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #Calculate GLCM with specified parameters
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for pixel pairs
    levels = 256  # Number of gray levels
    symmetric = True
    normed = True
            
    glcm = graycomatrix(gray_image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)                

    cont = round(mean(graycoprops(glcm, 'contrast').ravel()),4)
    diss = round(mean(graycoprops(glcm, 'dissimilarity').ravel()),4)
    homo = round(mean(graycoprops(glcm, 'homogeneity').ravel()),4)
    ener = round(mean(graycoprops(glcm, 'energy').ravel()),4)
    corr = round(mean(graycoprops(glcm, 'correlation').ravel()),4)
    asm = round(mean(graycoprops(glcm, 'ASM').ravel()),4)

    frequencies = [0.1, 0.3, 0.5]
    kernels = []

    # Generate Gabor filter kernels
    for frequency in frequencies:
        for theta in angles:
            kernel = np.real(gabor(gray_image, frequency, theta=theta))
            kernels.append(np.mean(kernel))

    # Convert the list of Gabor features to a numpy array
    gabor_features = np.array(kernels)

    # Convert the cropped image to CMYK color space manually
    b, g, r = cv2.split(img)
    c = 255 - r
    m = 255 - g
    y = 255 - b
    k = np.minimum(np.minimum(c, m), y)
    c = cv2.subtract(c, k)
    m = cv2.subtract(m, k)
    y = cv2.subtract(y, k)

    # Compute the average value of the K channel
    k_value = np.mean(k)

    data = [{'Basename':base_name,
             'Contrast': cont, 'Dissimilarity': diss, 'Homogeneity':homo, 
             'Energy':ener, 'Correlation':corr,'ASM':asm,
             'Gabor1':round(gabor_features[0],4),'Gabor2':round(gabor_features[1],4),'Gabor3':round(gabor_features[2],4),
             'Gabor4':round(gabor_features[3],4),'Gabor5':round(gabor_features[4],4),'Gabor6':round(gabor_features[5],4),
             'Gabor7':round(gabor_features[6],4),'Gabor8':round(gabor_features[7],4),'Gabor9':round(gabor_features[8],4),
             'Gabor10':round(gabor_features[9],4),'Gabor11':round(gabor_features[10],4),'k-Value':round(k_value,4),
            }]

    header_names = ['Basename','Contrast', 'Dissimilarity','Homogeneity','Energy','Correlation', 'ASM', 
                    'Gabor1','Gabor2','Gabor3','Gabor4','Gabor5','Gabor6','Gabor7','Gabor8',
                    'Gabor9','Gabor10','Gabor11','k-Value']

    csv_file_path = 'data.csv'
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_names)
        
        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()
        
        # Write rows
        for row in data:
            writer.writerow(row)

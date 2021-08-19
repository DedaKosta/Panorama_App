import numpy as np
import cv2 as cv
import os

MIN_MATCH_COUNT = 10

def UcitajSlike(dir):
    images = []
    for filename in os.listdir(dir):
        images.append(os.path.join(dir, filename))
    return images

################################################################

def NadjiPar(sl1, sl2, pisi):
    if(pisi == False):
        img1 = cv.imread(sl1, cv.IMREAD_GRAYSCALE)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)

        img2 = cv.imread(sl2, cv.IMREAD_GRAYSCALE)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        if (len(matches) == 0):
            return False, -1, ""

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M,_ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            warped_image1 = warpImages(img2, img1, M)

            filename = ".\Slike\output.jpg"
            filename.replace('"', "'")
            cv.imwrite(filename, warped_image1)
            return True, j, ".\Slike\output.jpg"
        else:
            return False, -1, ""
    else:
        return False, -1, ""

################################################################

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img

################################################################

slike = []
slike = UcitajSlike('.\Slike')

slikeCpy = []

pronadjenPar = False
imaPar = False
parId = 0
newString = ""

while len(slike) != 1:
    slikeCpy = []
    pronadjenPar = False
    imaPar = False
    parId = 0
    newString = ""
    for i in range(len(slike)):
        for j in range(i + 1, len(slike)):
            if(slike[i] == "" or slike[j] == ""):
                continue
            else:
                imaPar, parId, newString = NadjiPar(slike[i], slike[j], pronadjenPar)
                if(imaPar == False):
                    slikeCpy.append(slike[j])
                else:
                    slikeCpy.append(newString)
                    slike[j] = ""
                    pronadjenPar = True
        if(pronadjenPar == False):
            slikeCpy.append(slike[i])
    slike = []
    for sl in slikeCpy:
        slike.append(sl)

finale = cv.imread(slike[0])
cv.imshow("Output", finale)

cv.waitKey(0)
cv.destroyAllWindows()
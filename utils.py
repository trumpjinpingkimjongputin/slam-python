import cv2
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        self.GX = 16//2
        self.GY = 12//2

        self.orb = cv2.ORB_create(1000) 
    def extract(self, img):

        # cv2.detect had some problems here - features were only visible within small grid like spaces
        """
        sy = img.shape[0]//self.GY
        sx = img.shape[1]//self.GX
        akp = []
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                # keypoints and descriptors
                img_chunk = img[ry:ry+sy , rx: rx+sx]
                # print(img_chunk.shape)
                # kp, des = self.orb.detectAndCompute(img_chunk,None)
                kp = self.orb.detect(img_chunk, None)
                for p in kp:
                    print(p.pt)
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    # u, v = map(lambda x: int(round(x)), p.pt)
                    # cv2.circle(img,(u,v), color=(0,255,0), radius = 3)
                    akp.append(p)
        return akp
        """

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        return feats
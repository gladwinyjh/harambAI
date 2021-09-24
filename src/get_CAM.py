import numpy as np
import cv2


def normalize(image):
    image = image - np.min(image) 
    image = image / np.max(image)
    image = np.uint8(255 * image)
    return image


def getCAM(feature_maps, weight_softmax):
    batch_size, num_channels, height, width = feature_maps.shape
    cam = weight_softmax.dot(feature_maps.reshape((num_channels, height*width)))
    # ReLU
    cam[cam<0] = 0
    cam = cam.reshape(height, width)
    cam_img = normalize(cam)
    return cam_img


def generate(image, model, idx, feature_maps, device):
    # Get softmax weight w.r.t the target class idx
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    # Get CAM
    CAM = getCAM(feature_maps[0].cpu(), weight_softmax[idx])
    # Resize heatmap to same as input image, then apply jet colormap
    # Note that cv2 will apply it in reversed because it reads images in BGR
    heatmap = cv2.applyColorMap(cv2.resize(CAM,(image.size[0], image.size[1])),
                                cv2.COLORMAP_JET)
    # Convert to BGR heatmap to RGB
    heatmap =  cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Superimpose heatmap * (alpha) hyper-parameter with input image
    # Alpha set to 0.4 by default             
    superimposed_img = heatmap * 0.4 + image
    superimposed_img = normalize(superimposed_img)
    return superimposed_img
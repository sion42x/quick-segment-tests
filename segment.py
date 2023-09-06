import torch
import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def isolate_segmentation(index_):
  global segmented_area
  sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
  print('the segmentation area has {} pixels'.format(sorted_anns[index_]['area']))
  mask = np.expand_dims(sorted_anns[index_]['segmentation'],axis=-1)
  segmented_area = (image*mask).astype('uint8')
  segmented_area[segmented_area==0]=1
  segmented_area *=255
  imsave('mask.png',segmented_area)
  plt.imshow(segmented_area)
  plt.axis('off')
  plt.show()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint="./sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

mask_annotator = sv.MaskAnnotator()

image_bgr = cv2.imread('./image.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def visualize_segmentations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    text_color = (0,0, 0) # black color
    text_scale = 1
    text_thickness = 1
    for j,ann in enumerate(sorted_anns):
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        # Compute the center of mass of the binary mask
        text_size, _ = cv2.getTextSize(str(j), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        M = cv2.moments(m.astype('uint8'))
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        text_x = center_x - text_size[0]//2
        text_y = center_y + text_size[1]//2
        for i in range(3):
            img[:,:,i] = color_mask[i]
        cv2.putText(img, str(j), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
        img = np.dstack((img, m*0.35))
        plt.imshow(img)

masks = mask_generator.generate(image_rgb)
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image_bgr, detections)
cv2.imwrite('./output.png', annotated_image)

plt.figure(figsize=(10,10))
plt.imshow(image_rgb)
visualize_segmentations(masks)
plt.axis('off')
plt.savefig('./outputann.png')

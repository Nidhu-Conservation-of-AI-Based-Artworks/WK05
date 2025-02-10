import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from transformers import DetrImageProcessor


def interpolate(input, size):
  return F.interpolate(input, size, mode="bicubic", align_corners=False)

class DepthAnythingPostProcessor:
  @classmethod
  def process_output(cls, output, image):
    prediction = interpolate(output["predicted_depth"].unsqueeze(1), image.size[::-1])
    pixel_values = prediction.squeeze().cpu().detach().numpy()
    pixel_values255 = (pixel_values * 255 / np.max(pixel_values)).astype("uint8")
    return Image.fromarray(pixel_values255)

class DetrPostProcessor:
  OBJ_MODEL = "facebook/detr-resnet-101"
  processor = None

  @classmethod
  def process_output(cls, output, image, id2label, confidence=0.9):
    if cls.processor == None:
      cls.processor = DetrImageProcessor.from_pretrained(DetrPostProcessor.OBJ_MODEL, revision="no_timm")
    results = cls.processor.post_process_object_detection(output, confidence, [image.size[::-1]])[0]

    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
      bbox = box.tolist()
      draw.rectangle(tuple(bbox), outline=(0, 255, 0), width=3)
      draw.text((bbox[0], bbox[1]-16), id2label[label.item()], fill=(0, 255, 0), font_size=16)
      draw.text((bbox[0], bbox[1]), f"{round(score.item(), 3)}", fill=(255, 0, 255), font_size=16)
    return output_image

class SegformerPostProcessor:
  ADE20K_PALETTE = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]

  @classmethod
  def process_output(cls, output, image):
    logits = interpolate(output.logits, image.size[::-1])
    predictions = torch.argmax(logits, dim=1) + 1
    seg_img = Image.fromarray(predictions.squeeze().cpu().numpy().astype("uint8"))
    seg_img.putpalette(SegformerPostProcessor.ADE20K_PALETTE)
    return seg_img.convert("RGB")

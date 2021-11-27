from PIL import Image

im = Image.open('sample/sample_new.png')
im = im.crop([0,0,800,400])
im.save('VDSR_crop.png')
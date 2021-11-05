from PIL import Image

im = Image.open('sample/demo2.jpg')
print(im)
im.close()
print(im)
im.show()
import numpy as np 
from PIL import Image,ImageDraw, ImageFont

def getwidth(t):
	if t.kids==[]:
		return 1
	return getwidth(t.kids[0])+getwidth(t.kids[1])

def getdepth(t):
	if t.kids==[]:
		return 0
	return max(getdepth(t.kids[0]),getdepth(t.kids[1]))+1

def drawnode(draw_img,trained_tree,x,y):

	fnt = ImageFont.truetype("arial.ttf",size=18)
	if trained_tree.class_==None:
		w1=getwidth(trained_tree.kids[0])*50
		w2=getwidth(trained_tree.kids[1])*50

		left=x-(w1+w2)/2
		right=x+(w1+w2)/2

		draw_img.text((x-20,y-10),str(trained_tree.op),(0,0,0),font=fnt)

		draw_img.line((x,y,left+w1/2,y+100),fill=(255,0,0))
		draw_img.line((x,y,right-w2/2,y+100),fill=(255,0,0))

		drawnode(draw_img,trained_tree.kids[0],left+w1/2,y+100)
		drawnode(draw_img,trained_tree.kids[1],right-w2/2,y+100)

	else:
		draw_img.text((x-10,y),str(trained_tree.class_),(0,0,0),font=fnt)
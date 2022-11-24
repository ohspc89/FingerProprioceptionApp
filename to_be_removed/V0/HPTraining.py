import math


from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Quad
from random import randrange
from kivy.core.window import Window


Window.size = (800, 600)

class FloatLayout(FloatLayout):

	
	
	def damnwrong(self):
        # Color values are set randomly
        # This may cause the two colors become less distinguishable
        # Should be some sort of limit in setting the numbers
		threshold_left = self.ids.cw.width*2.0/3.0
		threshold_right = self.ids.cw.width/3.0
		rgb_index1 = randrange(0 , 5, 1)
		rgb_index2 = randrange(0 , 5, 1)
		rgblist1 = [(0.1 , 0, 0.4 , 1),(0.1 , 0, 0.9 , 1),(0.1 , 0.2 , 0.2 , 1),(0.4 , 0, 0.1 , 1),(0.4 , 0, 0.6 , 1)]
		rgblist2 = [(0.3 , 1, 0.7 , 1),(0.2 , 0.9 , 0.9 , 1),(0.9 , 1 , 0.1 , 1),(1 , 0.4, 0.6 , 1),(0.8 , 0.9 , 1 , 1)]
		self.ids.cw.bg_color_after = rgblist1[rgb_index1]
		self.ids.cw.bg_color_before = rgblist2[rgb_index2]
		self.ids._more_left.background_normal = ''
		self.ids._more_left.background_color = self.ids.cw.bg_color_after
		self.ids._more_right.background_normal = ''
		self.ids._more_right.background_color = self.ids.cw.bg_color_before
        # Moving to the left by 1 degree

		self.ids.cw.degree = self.ids.cw.degree + 1
		if self.ids.cw.degree >= 0:
			temp_degree = self.ids.cw.degree
			translate_length = self.ids.cw.height*math.tan(temp_degree*math.pi/180)
			#print(temp_degree)
			if translate_length >= threshold_left:
				self.ids.cw.degree = self.ids.cw.degree - 1
				self.ids.cw.tl = threshold_left
			else:
				self.ids.cw.tl = 0
				self.ids.cw.tl = self.ids.cw.tl + translate_length
		else:
			temp_degree = -self.ids.cw.degree
			translate_length = self.ids.cw.height*math.tan(temp_degree*math.pi/180)
			if translate_length >= threshold_right:
				self.ids.cw.degree = self.ids.cw.degree + 1
				self.ids.cw.tl = -threshold_right
			else:
				self.ids.cw.tl = 0
				self.ids.cw.tl = self.ids.cw.tl - translate_length
			

	def damnright(self):
		threshold_left = self.ids.cw.width*2.0/3.0
		threshold_right = self.ids.cw.width/3.0
		rgb_index1 = randrange(0,5,1)
		rgb_index2 = randrange(0,5,1)
		rgblist1 = [(0.1 , 0, 0.4 , 1),(0.1 , 0, 0.9 , 1),(0.1 , 0.2 , 0.2 , 1),(0.4 , 0, 0.1 , 1),(0.4 , 0, 0.6 , 1)]
		rgblist2 = [(0.3 , 1, 0.7 , 1),(0.2 , 0.9 , 0.9 , 1),(0.9 , 1 , 0.1 , 1),(1 , 0.4, 0.6 , 1),(0.8 , 0.9 , 1 , 1)]
		self.ids.cw.bg_color_after = rgblist1[rgb_index1] 
		self.ids.cw.bg_color_before = rgblist2[rgb_index2]
		self.ids._more_right.background_normal = ''
		self.ids._more_right.background_color = self.ids.cw.bg_color_before
		self.ids._more_left.background_normal = ''
		self.ids._more_left.background_color = self.ids.cw.bg_color_after
        # Moving to the right by 1 degree
		
		self.ids.cw.degree = self.ids.cw.degree - 1
		if self.ids.cw.degree >= 0:
			temp_degree = self.ids.cw.degree
			translate_length = self.ids.cw.height*math.tan(temp_degree*math.pi/180)
			#print(temp_degree)
			if translate_length >= threshold_left:
				self.ids.cw.degree = self.ids.cw.degree - 1
				self.ids.cw.tl = threshold_left
			else:
				self.ids.cw.tl = 0
				self.ids.cw.tl = self.ids.cw.tl + translate_length
		else:
			temp_degree = -self.ids.cw.degree
			translate_length = self.ids.cw.height*math.tan(temp_degree*math.pi/180)
			if translate_length >= threshold_right:
				self.ids.cw.degree = self.ids.cw.degree + 1
				self.ids.cw.tl = -threshold_right
			else:
				self.ids.cw.tl = 0
				self.ids.cw.tl = self.ids.cw.tl - translate_length

class ProprioceptiveApp(App):

    def build(self):
        return FloatLayout()

if __name__ == '__main__':
    ProprioceptiveApp().run()

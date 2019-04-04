from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Quad
from random import random


class FloatLayout(FloatLayout):
    def damnwrong(self):
        # Color values are set randomly
        # This may cause the two colors become less distinguishable
        # Should be some sort of limit in setting the numbers
        self.ids.cw.bg_color_after = (random(), random(), random(), 1) 
        self.ids.cw.bg_color_before = (random(), random(), random(), 1)
        # Moving to the left by 0.1
        # This may need more discussion
        self.ids.cw.lr = self.ids.cw.lr - 0.1

    def damnright(self):
        self.ids.cw.bg_color_after = (random(), random(), random(), 1) 
        self.ids.cw.bg_color_before = (random(), random(), random(), 1)
        # Moving to the right by 0.1
        self.ids.cw.lr = self.ids.cw.lr + 0.1

class ProprioceptiveApp(App):

    def build(self):
        return FloatLayout()

if __name__ == '__main__':
    ProprioceptiveApp().run()

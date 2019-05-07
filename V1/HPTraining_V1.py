import math
from random import randrange
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.properties import ObjectProperty

Builder.load_string("""
#:import math math

# Intersecting gridlines for calibration
<GridLines@Widget>:
    # vertical line
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.x, self.y
            size: self.right, self.top
        Color:
            rgba: 0, 0, 0, 1
        Line:
            points: self.center_x, self.y, self.center_x, self.top
    # horizontal line
    canvas.after:
        Color:
            rgba: 0, 0, 0, 1
        Line:
            points: self.x, self.center_y, self.right, self.center_y

# Button size is proportional to the screen size
<CustomButton@Button>
    size_hint: 0.3, 0.05
    spacing: 10

<ColorScreen@Widget>
    # Initial screen color on the right: Blue 
    bg_color_before: 0, 0, 1, 1
    canvas.before:
        Color:
            rgba: self.bg_color_before
        Rectangle:
            pos: self.x, self.y 
            size: self.width, self.height
    # Initial screen color on the left: Red
    bg_color_after: 1, 0, 0, 1

    # initial info??
    # This degree should be determined after the position of MP joint is set.
    # left: < 0; right: > 0
    degree: -45 
    canvas.after:
        Color:
            rgba: self.bg_color_after
        Quad:
            # there may need further discussion regarding these points
            # current setting of the MP joint = 0.7 * total width of the colored screen
            points: self.x, self.y, self.x, self.top, self.right*0.7 + self.height*math.tan(math.radians(self.degree)), self.top, self.right*0.7, self.y
            
<CalibrationScreen>:
    FloatLayout:
        GridLines:
            on_touch_down:
                root.manager.current = "screen_two"

<CustomPopup>:
    size_hint: .5, .5
    auto_dismiss: false
    title: "Popup Window"
    Button:
        text: "Close"
        on_press: root.dismiss()

<ParamInputScreen>:
    pid_text_input: pid
    flen_text_input: flen
    fwid_text_input: fwid
    initd_text_input: initd
    BoxLayout:
        orientation: "vertical"

        canvas:
            Color:
                rgba: 1, 1, 1, 0.95
            Rectangle:
                size: self.right, self.top
        
        RelativeLayout:

            Label:
                color: 0, 0, 0, 1
                text: "Participant number"
                pos_hint:{'center_x':0.2, 'center_y':0.5}

            TextInput:
                id: pid
                font_size: 20
                multiline: False 
                size_hint: (0.1, 0.25)
                pos_hint:{'center_x': 0.7, 'center_y':0.5}
                focus: True

        RelativeLayout:

            Label:
                color: 0, 0, 0, 1
                text: "Initial change(deg)"
                pos_hint:{'center_x':0.2, 'center_y':0.5}

            TextInput:
                id: initd
                font_size: 20
                multiline: False 
                size_hint: (0.1, 0.25)
                pos_hint:{'center_x': 0.7, 'center_y':0.5}
                focus: True

        RelativeLayout:

            Label:
                color: 0, 0, 0, 1 
                text: "Dominant hand"
                pos_hint: {'center_x':0.2}

            CheckBox:
                color: 0, 0, 0, 1
                group: "dominant_hand"
                value: root.left
                pos_hint: {'center_x':.6}
                size_hint_x: 0.1
                on_active: root.if_active_l(self.active)

            Label:
                color: 0, 0, 0, 1
                text: "Left"
                pos_hint: {'center_x':.65}

            CheckBox:
                id: righty
                color: 0, 0, 0, 1
                group: "dominant_hand"
                value: root.right
                pos_hint: {'center_x':.75}
                size_hint_x: 0.1
                on_active: root.if_active_r(self.active)

            Label:
                color: 0, 0, 0, 1
                text: "Right"
                pos_hint: {'center_x':.8}

        RelativeLayout:

            Label:
                color: 0, 0, 0, 1
                text: "Finger width(mm)"
                pos_hint:{'center_x':0.2, 'center_y':0.5}

            TextInput:
                id: fwid
                font_size: 20
                multiline: False 
                size_hint: (0.1, 0.25)
                pos_hint:{'center_x': 0.7, 'center_y':0.5}
                focus: True

        RelativeLayout:

            Label:
                color: 0, 0, 0, 1
                text: "Finger length(mm)"
                pos_hint:{'center_x':0.2, 'center_y':0.5}

            TextInput:
                id: flen
                font_size: 20
                multiline: False 
                size_hint: (0.1, 0.25)
                pos_hint:{'center_x': 0.7, 'center_y':0.5}
                focus: True

        Button:
            text: "Next"
            color: 1, 1, 1, 1
            font_size: 20
            size_hint: (.2, .2)
            pos_hint: {'center_x':0.5, 'y':self.top}
            on_press: 
                root.manager.current = "screen_three"
                root.assign_variables()

        Label:
            text: ""
            size_hint: (.1, .1)

# The actual color screen
<TestScreen>:
    RelativeLayout:
        CustomButton:
            id: _more_left 
            background_color: 1, 0, 0, 1
            background_normal: ""
            pos_hint: {'x':0.05, 'y':0.05}
            on_press: root.on_the_left()

        CustomButton:
            id: _more_right
            background_color: 0, 0, 1, 1
            background_normal: ""
            pos_hint: {'x':0.38, 'y':0.05}
            on_press: root.on_the_right()

        ColorScreen:
            id: cw
            # have the colorscreen be centered!!
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            size_hint: 0.9, 0.7
            # have this property to have an access to it all the time
            quad_points: self.x, self.y, self.x, self.top, self.right*0.7 + self.height*math.tan(math.radians(self.degree)), self.top, self.right*0.7, self.y

<screen_manager>:
    CalibrationScreen:
        id: calibsc
    ParamInputScreen:
        id: paramsc
        name: "screen_two"
    TestScreen:
        id: testsc
        name: "screen_three"
""")

class CalibrationScreen(Screen):
    pass

class ParamInputScreen(Screen):

    right = ObjectProperty(True)
    left = ObjectProperty(False)

    def assign_variables(self):
        self.pid = self.pid_text_input.text
        self.flen = self.flen_text_input.text
        self.fwid = self.fwid_text_input.text
        self.initd = self.initd_text_input.text
        print(self.pid, self.flen, self.fwid, self.initd)

    def if_active_r(self, state):
        if state:
            print(self.right)
       
    def if_active_l(self, state):
        if state:
            print("It's left")

class TestScreen(Screen):

    def __init__(self, **kwargs):
        super(TestScreen, self).__init__(**kwargs)
        self.rgblist1 = [(1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
        self.rgblist2 = [(0, 0, 1, 1), (0.5, 0, 1, 1), (1, 0.56, 0.75, 1)]
        # counting the number of reverses to terminate the study
        self.rev_count = 0
        # checking if the reverse is happening
        self.prev_choice = list()
        # This amount needs more discussion.
        # Currently it starts at 10 and divided by 2 upon reversal
        self.delta_d = 10

    def deltadeg(self):
        return (self.parent.ids.paramsc.initd)


    # changes the color of the buttons as well as the screen
    def change_col_setting(self):
        rgb_index = randrange(0, 3, 1)
        self.ids.cw.bg_color_after = self.rgblist1[rgb_index]
        self.ids.cw.bg_color_before = self.rgblist2[rgb_index]
        self.ids._more_left.background_normal = ''
        self.ids._more_left.background_color = self.ids.cw.bg_color_after
        self.ids._more_right.background_normal = ''
        self.ids._more_right.background_color = self.ids.cw.bg_color_before

    # keep track of reversals
    def track_rev(self, response):
        if len(self.prev_choice) > 0:
            if self.prev_choice[-1] != response:
                self.rev_count += 1
                #self.delta_d = self.delta_d/2
        self.prev_choice.append(response)

    def update_delta_d(self):
        self.delta_d = float(self.parent.ids.paramsc.initd)
        if self.rev_count > 0:
            self.delta_d = self.delta_d / (2.0**self.rev_count)

    def on_the_left(self):

        self.change_col_setting()

        self.track_rev('otl')
        
        self.update_delta_d()

        temp_l = self.ids.cw.degree - self.delta_d
        if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp_l)) < self.ids.cw.x):
            self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.x - self.ids.cw.quad_points[6]) / self.ids.cw.height))
        else:
            self.ids.cw.degree = temp_l
        print(self.ids.cw.degree)


    def on_the_right(self):

        self.change_col_setting()

        self.track_rev('otr')

        self.update_delta_d()

        temp_r = self.ids.cw.degree + self.delta_d
        if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp_r)) > self.ids.cw.right):
            self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.right - self.ids.cw.quad_points[6]) / self.ids.cw.height))
        else:
            self.ids.cw.degree = temp_r 
        print(self.ids.cw.degree)
            

#screen_manager = ScreenManager(transition=FadeTransition())
#screen_manager.add_widget(CalibrationScreen(name="screen_one"))
#screen_manager.add_widget(ParamInputScreen(name="screen_two"))
#screen_manager.add_widget(TestScreen(name="screen_three"))

class screen_manager(ScreenManager):
    pass
#    def __init__(self, **kwargs):
#        super(screen_manager, self).__init__(**kwargs)
#        self.add_widget(CalibrationScreen(name="screen_one"))
#        self.add_widget(ParamInputScreen(name="screen_two"))
#        self.add_widget(TestScreen(name="screen_three"))

class ProprioceptiveApp(App):
    def build(self):
        return screen_manager(transition=FadeTransition())

if __name__ == '__main__':
    ProprioceptiveApp().run()

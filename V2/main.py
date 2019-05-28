import math
from random import randrange
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.storage.jsonstore import JsonStore
import time

# Get the start time of the experiment; will use this as the file name
timestamp = time.strftime("%Y%m%d_%H:%M:%S")
store = JsonStore(".".join([timestamp, 'json']))

class CalibrationScreen(Screen):

    # Popup window
    def show_popup(self):

        the_popup = CalibPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))
        the_popup.open()

class CalibPopup(Popup):
    pass

class ParamPopup(Popup):
    pass

class ParamInputScreenOne(Screen):

    male = ObjectProperty(True)
    female = ObjectProperty(False)
    right = ObjectProperty(True)
    left = ObjectProperty(False)

    gender = ObjectProperty(None)
    handed_chk = ObjectProperty(False)

    # Popup window
    def show_popup(self):

        the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        # Check if any of the parameter inputs is missing!
        if any([self.pid == "", self.age == "", self.gender == None, self.handed_chk == False]) is True:
            the_popup.argh.text = "Something's missing!"
        else:
            store.put("SUBJ_info", subid = self.pid, age = self.age, gender = self.gender, right_used = self.ids.rightchk.active)
        the_popup.open()

    def assign_variables(self):
        self.pid = self.pid_text_input.text
        self.age = self.age_text_input.text
        # subject information
        print("pid:", self.pid, "age:", self.age, "gender:", self.gender, "Right_Dominant:", self.ids.rightchk.active)


    def if_active_m(self, state):
        if state:
            # Whill change the orientation of the testscreen's colorscreen
            self.gender = "M"

    def if_active_f(self, state):
        if state:
            self.gender = "F"

    def if_active_r(self, state):
        if state:
            # Whill change the orientation of the testscreen's colorscreen
            self.parent.ids.testsc.handedness.dir = 1
            #self.parent.ids.testsc.handedness.degree = -35

            # Just for fool-proof
            self.handed_chk = True

    def if_active_l(self, state):
        if state:
            self.parent.ids.testsc.handedness.dir = -1
            #self.parent.ids.testsc.handedness.degree = 35

            # Just for fool-proof
            self.handed_chk = True

class ParamInputScreenTwo(Screen):

    # Popup window
    def show_popup2(self):

        the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        # Check if any of the parameter inputs is missing!
        if any([self.flen == "", self.fwid == "", self.initd == "", self.mprad == ""]):
            the_popup.argh.text = "Something's missing!"
        else:
            store.put("ANTH_measures", flen = self.flen, fwid = self.fwid, init_step = self.initd, MPJR = self.mprad)
        the_popup.open()

    def assign_variables(self):
        self.flen = self.flen_text_input.text
        self.fwid = self.fwid_text_input.text
        self.initd = self.initd_text_input.text
        self.mprad = self.mprad_text_input.text

        # subject anthropometric information
        print("finger length:", self.flen, "finger width:", self.fwid, "initial ss:", self.initd, "MP Joint Radius:", self.mprad)

        self.parent.ids.testsc.handedness.flen = self.flen
        self.parent.ids.testsc.handedness.mprad = self.mprad

class TestScreen(Screen):

    handedness = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(TestScreen, self).__init__(**kwargs)
        self.rgblist1 = [(1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
        self.rgblist2 = [(0, 0, 1, 1), (0.5, 0, 1, 1), (1, 0.56, 0.75, 1)]
        self.rgbindex = 0
        # counting the number of reverses to terminate the study
        self.rev_count = 0
        # checking if the reverse is happening
        self.prev_choice = list()
        # check the trial number
        self.trial_num = 0


    # changes the color of the buttons as well as the screen
    def change_col_setting(self):
        rgb_index = randrange(0, 3, 1)
        while rgb_index == self.rgbindex:
            rgb_index = randrange(0, 3, 1)
        self.ids.cw.bg_color_after = self.rgblist1[rgb_index]
        self.ids.cw.bg_color_before = self.rgblist2[rgb_index]
        self.ids._more_left.background_normal = ''
        self.ids._more_left.background_color = self.ids.cw.bg_color_after
        self.ids._more_right.background_normal = ''
        self.ids._more_right.background_color = self.ids.cw.bg_color_before
        self.rgbindex = rgb_index

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

        # Add the current choice, check if reversal is happening
        self.track_rev('otl')

        # Based on the updated reversal count, calculate the delta_d
        self.update_delta_d()

        # next step deviation angle
        temp_l = self.ids.cw.degree - self.delta_d

        # Set the left limit
        if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp_l)) < self.ids.cw.x):
            self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.x - self.ids.cw.quad_points[6]) / self.ids.cw.height))
        else:
            self.ids.cw.degree = temp_l

        store.put("_".join(["TRIAL", str(self.trial_num), "info"]), rev_cnt = self.rev_count, choice = self.prev_choice[-1], deg = self.ids.cw.degree, step_size = self.delta_d)

        self.trial_num += 1

        # Print the trial number and the deviation angle(deg)
        # The value of the deviation angle is the angle between
        # - the vertical line that passes the MP joint
        # - the line that connects the MP joint and the upper right point of the quadrilateral
        print(self.trial_num, self.ids.cw.degree)


    def on_the_right(self):

        self.change_col_setting()

        self.track_rev('otr')

        self.update_delta_d()

        temp_r = self.ids.cw.degree + self.delta_d

        # Set the right limit
        if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp_r)) > self.ids.cw.right):
            self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.right - self.ids.cw.quad_points[6]) / self.ids.cw.height))
        else:
            self.ids.cw.degree = temp_r

        store.put("_".join(["TRIAL", str(self.trial_num), "info"]), rev_cnt = self.rev_count, choice = self.prev_choice[-1], deg = self.ids.cw.degree, step_size = self.delta_d)

        self.trial_num += 1

        print(self.trial_num, self.ids.cw.degree)

class screen_manager(ScreenManager):
    pass

class ProprioceptiveApp(App):
    def build(self):
        return screen_manager(transition=FadeTransition())

if __name__ == '__main__':
    ProprioceptiveApp().run()

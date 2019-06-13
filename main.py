import math
from random import randrange
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.storage.jsonstore import JsonStore
import os, time
from kivy import platform

timestamp = time.strftime("%Y%m%d_%H:%M:%S")

# If running on an android device, set the right path to save the JSON file
if platform == 'android':
    from jnius import autoclass, cast, JavaException

    try:
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
    except JavaException:
        PythonActivity = autoclass('org.renpy.android.PythonActivity')


    Environment = autoclass('android.os.Environment')
    context = cast('android.content.Context', PythonActivity.mActivity)
    private_storage = context.getExternalFilesDir(Environment.getDataDirectory().getAbsolutePath()).getAbsolutePath()

    store = JsonStore(".".join([private_storage, timestamp, 'json']))

# This is mainly for testing on a Linux Desktop
else: 
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

    # Popup window to check if everything is saved properly
    def show_popup(self):

        the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        # Check if any of the parameter inputs is missing!
        if any([self.pid_text_input.text == "", self.age_text_input.text == "", self.gender == None, self.handed_chk == False]) is True:
            the_popup.argh.text = "Value Missing!" 
            the_popup.open()
        else:
            store.put("SUBJ_info", subid = self.pid_text_input.text, age = self.age_text_input.text, gender = self.gender, right_used = self.ids.rightchk.active)
            self.parent.current = "param_screen_two"

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

    # Popup window to check if everything is entered
    def show_popup2(self):

        the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        # Check if any of the parameter inputs is missing!
        if any([self.flen_text_input.text == "", self.fwid_text_input.text == "", self.initd_text_input.text == "", self.mprad_text_input.text == ""]):
            the_popup.argh.text = "Something's missing!"
            the_popup.open()
        else:
            store.put("ANTH_measures", flen = self.flen_text_input.text, fwid = self.fwid_text_input.text, init_step = self.initd_text_input.text, MPJR = self.mprad_text_input.text)

            # Give the mp joint radius input to draw the test screen display
            self.parent.ids.testsc.handedness.mprad = self.mprad_text_input.text
            self.parent.current = "test_screen"

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
        # session number
        self.session_num = 0


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
        #self.delta_d = float(self.parent.ids.paramsc.initd_text_input.text)
        self.delta_d = float(self.parent.ids.paramsc.initd_text_input.text) / (2.0**self.rev_count)

    def where_is_your_finger(self, rel_pos):

        # change the colors of the screen
        self.change_col_setting()

        # Add the current choice, check if reversal is happening
        self.track_rev(rel_pos)

        # Save the current degree
        degree_current = self.ids.cw.degree

        # Check if the respons('on the left' or 'on the right') is correct 
        # Get the current third x-coordinate of the quadrilateral, or the fourth point of the quadrilateral
        # Compare it with the true third x-coordinate of the quadrilateral
        # If the current x-coordinate is greater than the true value, the correct answer should be "left"
        # If the current x-coordinate is smaller than the true value, the correct answer should be "right"
        # If neither, the response is "on_the_spot"
        x_coord_current = self.ids.cw.quad_points[4]
        if x_coord_current > self.ids.cw.x_correct:
            correct_ans = "left"
        elif x_coord_current < self.ids.cw.x_correct:
            correct_ans = "right"
        else:
            correct_ans = "on_the_spot"

        # Based on the updated reversal count, calculate the delta_d
        self.update_delta_d()

        # next step deviation angle
        if rel_pos == 'left':
            temp = self.ids.cw.degree - self.delta_d

            # Set the left limit
            if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp)) < self.ids.cw.x):
                self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.x - self.ids.cw.quad_points[6]) / self.ids.cw.height))
            else:
                self.ids.cw.degree = temp

        elif rel_pos == 'right':
            temp = self.ids.cw.degree + self.delta_d

            # Set the right limit
            if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp)) > self.ids.cw.right):
                self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.right - self.ids.cw.quad_points[6]) / self.ids.cw.height))
            else:
                self.ids.cw.degree = temp

        store.put("_".join(["TRIAL", str(self.trial_num)]), session = self.session_num, rev_cnt = self.rev_count, choice = self.prev_choice[-1], deg = degree_current, step_size = self.delta_d, correct_x = self.ids.cw.x_correct, x_coord_current = x_coord_current, correct_ans = correct_ans, response_correct = rel_pos == correct_ans)

        self.trial_num += 1


        # Print the trial number and the deviation angle(deg)
        # The value of the deviation angle is the angle between
        # - the vertical line that passes the MP joint
        # - the line that connects the MP joint and the upper right point of the quadrilateral
        # print(self.trial_num, self.ids.cw.degree)

        # Switch screen if a participant has reached 20 trials or reversed 5 times
        if self.trial_num == 20 or self.rev_count == 5:
            if self.session_num == 1:
                self.parent.current = "outcome_screen"
            else:
                # A new session begins
                self.session_num = 1
                # Count the reverse from the beginning
                self.rev_count = 0
                # New display setting
                self.ids.cw.degree = -1 * self.ids.cw.dir * 55
                # There's no turning back
                self.ids.layout.remove_widget(self.ids._backward)
                # The buttons would be disabled until an experimenter presses the 'resume' button
                self.ids._more_left.disabled = True
                self.ids._more_right.disabled = True
            
        #if self.trial_num == 40 or self.rev_count == 10:

class OutcomeScreen(Screen):
    pass

class screen_manager(ScreenManager):
    pass

class ProprioceptiveApp(App):

    def build(self):
        return screen_manager(transition=FadeTransition())

if __name__ == '__main__':
    ProprioceptiveApp().run()

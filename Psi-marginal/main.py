import math, time, PsiMarginal
import numpy as np
from random import randrange
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.core.window import Window
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.storage.jsonstore import JsonStore
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

# Prepare dictionaries to save information
subj_info = {}
subj_anth = {}
subj_trial_info = {}

# These are Psi-Marginal Staircase related parameters
# mu = threshold parameter
# sigma = slope parameter
# StimLevels = delta angle
ntrials = 50
mu = np.linspace(0, 15, 61)
sigma = np.linspace(0.05, 1, 21)
lapse = np.linspace(0, 0.1, 15)
guessRate = 0.5
stimLevels = np.concatenate((np.arange(0, 10, 0.1), np.arange(10, 16, 1)))

thresholdPrior = ('normal', 13, 3)
slopePrior = ('gamma', 2, 0.3)
lapsePrior = ('beta', 2, 20)

Psi = PsiMarginal.Psi(stimLevels, Pfunction = 'Gumbel', nTrials = ntrials, threshold = mu, thresholdPrior = thresholdPrior, slope = sigma, slopePrior = slopePrior, guessRate = guessRate, guessPrior = ('uniform', None), lapseRate = lapse, lapsePrior = lapsePrior, marginalize = True)

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
            global subid
            subid = "_".join(["SUBJ", self.pid_text_input.text])
            global subj_info
            subj_info = {'age' : self.age_text_input.text, 'gender' : self.gender, 'right_used' : self.ids.rightchk.active}
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
            global subj_anth
            subj_anth = {'flen' : self.flen_text_input.text, 'fwid' : self.fwid_text_input.text, 'init_step' : self.initd_text_input.text, 'MPJR' : self.mprad_text_input.text}

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
        # checking if the reverse is happening
        self.prev_choice = list()
        # session number
        self.session_num = 0
        # check the trial number(within a session)
        self.trial_num = 0
        # Keep the record of total trials(regardless of session)
        self.trial_total = 0

        self.mov_angle = Psi.xCurrent

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
    def track_choices(self, response):
        self.prev_choice.append(response)

    def where_is_your_finger(self, rel_pos):

        # change the colors of the screen
        self.change_col_setting()

        # Add the current choice, check if reversal is happening
        self.track_choices(rel_pos)

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

        # Compare if the answer is correct
        right_or_wrong = int(rel_pos == correct_ans)
        global Psi
        Psi.addData(right_or_wrong)
        while Psi.xCurrent is None:
            pass

        # next step deviation angle
        if rel_pos == 'left':

            # Set the left limit
            if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(Psi.xCurrent)) < self.ids.cw.x):
                self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.x - self.ids.cw.quad_points[6]) / self.ids.cw.height))
            else:
                self.ids.cw.degree = float(Psi.xCurrent)

        elif rel_pos == 'right':

            # Set the right limit
            if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(Psi.xCurrent)) > self.ids.cw.right):
                self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.right - self.ids.cw.quad_points[6]) / self.ids.cw.height))
            else:
                self.ids.cw.degree = float(Psi.xCurrent)

        #global subj_trial_info
        subj_trial_info["_".join(["TRIAL", str(self.trial_total)])] = {'session': self.session_num, 'trial_in_session': self.trial_num, 'reference(deg)': self.ids.cw.false_ref, 'offset(deg)': degree_current, 'correct_x': self.ids.cw.x_correct, 'x_coord_current': x_coord_current, 'correct_ans': correct_ans, 'response': self.prev_choice[-1], 'response_correct': right_or_wrong}

        self.trial_num += 1
        self.trial_total += 1


        # Print the trial number and the deviation angle(deg)
        # The value of the deviation angle is the angle between
        # - the vertical line that passes the MP joint
        # - the line that connects the MP joint and the upper right point of the quadrilateral
        print("trial: ", self.trial_num, "session: ", self.session_num, "correct_ans: ", correct_ans, "rel_pos: ", rel_pos, "right_or_wrong: ", right_or_wrong, "Previous_delta_d: ", degree_current, "Next delta_d: ", self.ids.cw.degree, self.ids.cw.false_ref)

        if self.trial_num == 50:
            self.reset(self.session_num)

    def reset(self, session_num):
        # Renew the list of stored choices
        self.prev_choice = list()

        # Trial number renewed
        self.trial_num = 0

        # Psi marginal algorithm refreshed
        global Psi
        Psi = PsiMarginal.Psi(stimLevels, Pfunction = 'Gumbel', nTrials = ntrials, threshold = mu, thresholdPrior = thresholdPrior, slope = sigma, slopePrior = slopePrior, guessRate = guessRate, guessPrior = ('uniform', None), lapseRate = lapse, lapsePrior = lapsePrior, marginalize = True)

        # New display setting
        self.ids.cw.degree = float(Psi.xCurrent)

        if session_num == 0:
            # A new session begins
            self.session_num +=1

            # False reference moving to 45
            self.ids.cw.false_ref = 45
            # ... and the psi output will now be "added"
            self.ids.cw.degree_dir = 1

            # There's no turning back
            self.ids.layout.remove_widget(self.ids._backward)

            # The buttons would be disabled until an experimenter presses the 'resume' button
            self.ids._more_left.disabled = True
            self.ids._more_right.disabled = True

        # Only two sessions exist: 0 or 1
        # If session 1 finishes, you reset everthing to have a next subject
        else:
            # Dump everything to the store
            store.put(subid, subj_info = subj_info, subj_anth = subj_anth, subj_trial_info = subj_trial_info)

            self.session_num -= 1

            # False reference returning to 55
            self.ids.cw.false_ref = 55
            # ... and the psi output "subtracted"
            self.ids.cw.degree_dir = -1

            # Bring the back button again
            self.ids.layout.add_widget(self.ids._backward)

            # Total trial count reset to 0
            self.trial_total = 0

            # Go to the outcome screen
            self.parent.current = "outcome_screen"

class OutcomeScreen(Screen):

    def start_a_new_subject(self):
        self.parent.current = "param_screen_one"

class screen_manager(ScreenManager):
    pass

class ProprioceptiveApp(App):

    def build(self):
        return screen_manager(transition=FadeTransition())

if __name__ == '__main__':
    ProprioceptiveApp().run()

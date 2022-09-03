from manim import *
from gver import GVer
import numpy as np

class Human(Scene):
    def construct(self):
        p0 = [7, 4, 0]
        p1 = [-7,1, 0]
        p_list = [p0,p1]
        t_list = [0,1]
        times = 40
        for i in range(times):
            p, t = _next(p_list[i], p_list[i+1], time=t_list[i+1])
            p_list.append(p)
            t_list.append(t)
        dot = SVGMobject('src/human.svg',height=1).move_to(p0)

        gver = GVer()
        gver.add_updater(lambda m: m.watch(dot))

        self.add(gver)
        self.play(FadeIn(dot, run_time=.5))
        self.play(dot.animate(rate_func=rate_functions.linear, run_time=4).rotate(1.5*PI).move_to(p1))
        for i in range(2,times+2):
            p = p_list[i]
            t = t_list[i]
            self.add_sound('src/dunn.wav')
            self.play(dot.animate(rate_func=rate_functions.linear, run_time=t*4).rotate(1.5*PI*t).move_to(p))
        gver.clear_updaters()
        self.play(FadeOut(dot), gver.animate.stop_watch())
        gver.wink(self, 2)
        self.wait()

def _next(p0, p1, x_range=[-7,7], y_range=[-4,4], time=1):
    p0, p1 = np.array(p0), np.array(p1)
    dx, dy, dz = p1 - p0

    di_x = -1 if p1[0] in x_range else 1
    di_y = -1 if p1[1] in y_range else 1
    speed_x = dx * di_x / time
    speed_y = dy * di_y / time
    
    next_x = x_range[0] if speed_x < 0 else x_range[1]
    next_y = y_range[0] if speed_y < 0 else y_range[1]
    tx = (next_x - p1[0]) / speed_x
    ty = (next_y - p1[1]) / speed_y
    t = min(tx, ty)

    return p1 + np.array([t*speed_x, t*speed_y, 0]), t
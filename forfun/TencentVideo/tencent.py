from manim import *

class TencentVideo(Scene):
    def construct(self):
        _BLUE = '#10ABF2'
        _GREEN = '#7DE621'
        _ORANGE = '#FF8F21'
        bg = FullScreenRectangle()\
            .set_color('#90BDCB')\
            .set_stroke('#90BDCB')\
            .set_opacity(1)
        self.add(bg)

        logo = SVGMobject('src/tencent.svg', height=3).shift(DOWN*.5)
        inner = logo[3]
        mid = logo[2]
        outer_l = logo[1]
        outer_r = logo[0]
        title = VGroup(logo[4:]).center().shift(RIGHT)
        svg = VGroup(inner,mid,outer_l,outer_r)

        inner.set_color('#bbbbbb')
        mid.set_color('#cccccc')
        outer_l.set_color('#cccccc')
        outer_r.set_color('#dddddd')

        self.add_sound('src/sound.wav', time_offset=.1)
        self.play(
            bg.animate(run_time=1).set_color('#5CE7C6').set_stroke('#5CE7C6'),
            FadeIn(inner,   run_time=1, rate_func=timer(.3, .1)),
            FadeIn(mid,     run_time=1, rate_func=timer(.5, .1)),
            FadeIn(outer_l, run_time=1, rate_func=timer(.5, .1)),
            FadeIn(outer_r, run_time=1, rate_func=timer(.8, .1)),
        )
        self.play(
            bg.animate(run_time=2, rate_func=rate_functions.ease_in_expo)\
                .set_color(GRAY_A).set_stroke(GRAY_A),
            svg.animate(run_time=2, rate_func=timer(.4, .5)).scale(.6)
            )
        self.play(
            inner  .animate(run_time=.3).set_color(WHITE),
            mid    .animate(run_time=.3).set_color(_GREEN),
            outer_l.animate(run_time=.3).set_color(_ORANGE),
            outer_r.animate(run_time=.3).set_color(_BLUE),
        )
        self.play(
            svg.animate(run_time=.5).next_to(title, LEFT),
            GrowFromCenter(title, run_time=1, rate_func=timer(.1, .5))
        )
        self.wait(3)



# Deprecated! Use `atmm.time_manager` instead.
# install: pip install atmm
def timer(start=0, duration=1): 
    def rate_func(t, start=start, duration=duration):
        if   t < start:
            return 0.
        elif t > start+duration:
            return 1.
        else:
            return (t-start)/duration
    return rate_func
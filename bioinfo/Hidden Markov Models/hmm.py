from manim import *
from gver import GVer

#---------------------------------- compoments ----------------------------------

class GVGraph(VGroup):
    def __init__(
            self,
            table = [], 
            rows = 2,
            cols = 5,
            edges = 'right_like', #right_like (R + UR + DR) | downright_like (D + DR + R) | list of tuples of tuples e.g. [((0,0),(1,0)), ((0,1),(1,0))]
            v_buff = 1,
            h_buff = 1,
            circle_r = 0.3,
            arrow_thickness = 3,
            arrow_tip_length = 0.15,
            **kwargs
        ):
        super().__init__(**kwargs)
        if table:
            rows = len(table)
            cols = len(table[0])
        self._table = table
        self._rows = rows
        self._cols = cols
        self._edges = self._gen_edge(edges)
        self._v_buff = v_buff
        self._h_buff = h_buff
        self._circle_r = circle_r
        self._arrow_thickness = arrow_thickness
        self._arrow_tip_length = arrow_tip_length

        self.circles = self._draw_circles()
        self.arrows, self.arrows_dict = self._draw_arrows()
        self.numbers, self.numbers_dict = self._draw_numbers()
        self.add(self.circles, self.arrows, self.numbers)

    def _draw_circles(self):
        circles = [
            [
                Circle(radius=self._circle_r, fill_opacity=1).copy() for c in range(self._cols)
            ] for r in range(self._rows)
        ]
        circles = [VGroup(*c).arrange(RIGHT, buff=self._h_buff) for c in circles]
        circles = VGroup(*circles).arrange(DOWN, buff=self._v_buff)
        return circles
    def _draw_arrows(self):
        arrows = {}
        for edge in self._edges:
            start, end = edge
            start_pos = self.get_circle(start).get_center()
            end_pos = self.get_circle(end).get_center()
            arrows[edge] = Arrow(start_pos, end_pos, buff=self._circle_r+0.05, stroke_width=self._arrow_thickness, tip_length=self._arrow_tip_length)
        return VGroup(*arrows.values()), arrows
    def _draw_numbers(self):
        numbers = {}
        for r, row in enumerate(self._table):
            for c, item in enumerate(row):
                pos = (r,c)
                n = DecimalNumber(item, font_size=20, num_decimal_places=4)
                n.move_to(self.get_circle(pos))
                numbers[pos] = n
        return VGroup(*numbers.values()), numbers
        
    def _gen_edge(self, e):
        if type(e) is list:
            return e
        elif e == 'right_like':
            edges = [
                ((r,c),(i,c+1)) \
                    for c in range(self._cols-1) \
                        for r in range(self._rows) \
                            for i in range(self._rows)]
            return edges
        elif e == 'downright_like':
            edges = []
            for c in range(self._cols):
                for r in range(self._rows):
                    if c+1 < self._cols:
                        edges.append(((r,c),(r,c+1)))
                    if r+1 < self._rows:
                        edges.append(((r,c),(r+1,c)))
                    if c+1 < self._cols and r+1 < self._rows:
                        edges.append(((r,c),(r+1,c+1)))
            return edges
        else: return []
    def get_circle(self, pos):
        row = pos[0]
        col = pos[1]
        return self.circles[row][col]
    def get_arrow(self, pos):
        return self.arrows_dict[pos]
    def get_number(self, pos):
        return self.numbers_dict[pos]
    def get_arrows_start_with(self, pos):
        return [self.get_arrow(e) for e in self._edges if e[0]==pos]
    def get_arrows_end_with(self, pos):
        return [self.get_arrow(e) for e in self._edges if e[1]==pos]
    def set_row_fill(self, row, color):
        self.circles[row].set_fill(color)
        return self
    def set_col_stroke(self, col, color):
        for row in self.circles:
            for c, item in enumerate(row):
                if c == col:
                    item.set_stroke(color)
        return self
    def set_style(self, rows=None, cols=None, nums=None):
        if rows:
            for r,color in enumerate(rows):
                self.set_row_fill(r,color)
        if cols:
            for c,color in enumerate(cols):
                self.set_col_stroke(c, color)
        if nums:
            for r,color in enumerate(nums):
                for c in range(self._cols):
                    self.get_number((r,c)).set_color(color)
        return self
    def set_path_color(self, path, color=WHITE):
        for p in path:
            self.get_arrow(p).set_color(color).set_opacity(1)
        return self
    def hide_arrows(self):
        self.arrows.set_opacity(0.)
        return self
    def hide_numbers(self):
        self.numbers.set_opacity(0.)
        return self
    def show_number(self, pos):
        self.get_number(pos).set_opacity(1.)
        return self
    def animate_arrows_start_with(self, pos):
        arrows = self.get_arrows_start_with(pos)
        return [GrowArrow(a.set_color(YELLOW).set_opacity(1.)) for a in arrows]
    def animate_arrows_end_with(self, pos):
        arrows = self.get_arrows_end_with(pos)
        return [GrowArrow(a.set_color(YELLOW).set_opacity(1.)) for a in arrows]
    def animate_path(self, path):
        return [ Indicate(self.get_arrow(p)) for p in path ]
        
    # @staticmethod
    # def flatten(l):
    #     return [item for sublist in l for item in sublist]
    def test(self):
        self.circles.set_fill(YELLOW)
    
class GVNode(VGroup):
    def __init__(self, content, **kwargs):
        super().__init__(**kwargs)
        self.circle = Circle(radius=0.3, fill_opacity=1.)
        self.tex = Tex(content)
        self.add(self.circle, self.tex)
    def set_fill_(self, color):
        self.circle.set_fill(color)
        return self
    def set_stroke_(self, color):
        self.circle.set_stroke(color)
        return self
    def set_color_(self, color):
        self.tex.set_color(color)
        return self

class HmmDiagram(VGroup):
    def __init__(
        self,
        states = [],
        symbols = [],
        transition_prob = [],
        emission_prob = [],
        **kwargs
    ):
        super().__init__(**kwargs)
        self._states = states
        self._symbols = symbols
        self._transition = transition_prob
        self._emission = emission_prob

        self.states = self._draw_states()
        self.symbols = self._draw_symbols()
        _nodes = VGroup(self.states, self.symbols).arrange(RIGHT, buff=2.)
        self.arrows, self.arrows_dict = self._draw_arrows()
        self.add(_nodes, self.arrows)
        if self._transition and self._emission:
            self.probs, self.probs_dict = self._draw_probs()
            self.add(self.probs)

    def _draw_node(self, content):
        return GVNode(content)
    def _draw_states(self):
        return VGroup(*[self._draw_node(st) for st in self._states]).arrange(DOWN, buff=2.)
    def _draw_symbols(self):
        return VGroup(*[self._draw_node(sy) for sy in self._symbols]).arrange(DOWN, buff=1.)
    def _draw_arrows(self):
        arrows = {}
        # transition
        for idx1, st1 in enumerate(self._states):
            for idx2, st2 in enumerate(self._states):
                if st1 == st2:
                    arrows[st1 + st2] = CurvedArrow(
                        self.get_state(st1).get_bottom(), 
                        self.get_state(st2).get_top(), 
                        angle=-TAU*0.8,
                        stroke_width=3, 
                        tip_length=0.3,
                    )
                else:
                    if idx1 < idx2:
                        arrows[st1 + st2] = CurvedArrow(
                            self.get_state(st1).get_bottom(), 
                            self.get_state(st2).get_top(), 
                            stroke_width=3, 
                            tip_length=0.3,
                        )
                    else:
                        arrows[st1 + st2] = CurvedArrow(
                            self.get_state(st1).get_top(), 
                            self.get_state(st2).get_bottom(), 
                            stroke_width=3, 
                            tip_length=0.3,
                        )
        # emission
        for st in self._states:
            for sy in self._symbols:
                arrows[st + sy] = Arrow(
                    self.get_state(st),
                    self.get_symbol(sy),
                    buff=.05,
                    stroke_width=3,
                    tip_length=0.3,
                )
        return VGroup(*arrows.values()), arrows
    def _draw_probs(self):
        _probs = {}
        # transition
        for idx1, st1 in enumerate(self._states):
            for idx2, st2 in enumerate(self._states):
                _probs[st1 + st2] = self._transition[idx1][idx2]
        # emission
        for idx1, st in enumerate(self._states):
            for idx2, sy in enumerate(self._symbols):
                _probs[st + sy] = self._emission[idx1][idx2]
        ##
        _up_arrows = []
        for i in range(1, len(self._states)):
            for j in range(i):
                _up_arrows.append(self._states[i] + self._states[j])
        probs = {}
        for name, value in _probs.items():
            if name in _up_arrows:
                probs[name] = DecimalNumber(value, font_size=24).next_to(self.get_arrow(name).get_right(), LEFT*.5)
            else:
                probs[name] = DecimalNumber(value, font_size=24).next_to(self.get_arrow(name).get_center(), LEFT*.6)
        return VGroup(*probs.values()), probs
    def get_state(self, name):
        idx = self._states.index(name)
        return self.states[idx]
    def get_symbol(self, name):
        idx = self._symbols.index(name)
        return self.symbols[idx]
    def get_arrow(self, name):
        return self.arrows_dict[name]
    def get_prob(self, name):
        return self.probs_dict[name]
    def hide_arrows_and_probs(self):
        self.arrows.set_stroke(opacity=0.)
        for a in self.arrows:
            a.get_tip().set_fill(opacity=0.)
        self.probs.set_opacity(0.)
        return self
    def show_arrow(self, name, color=WHITE):
        self.get_arrow(name).set_color(color).set_stroke(opacity=1.).get_tip().set_fill(opacity=1.)
        return self.get_arrow(name)
    def show_prob(self, name, color=WHITE):
        self.get_prob(name).set_opacity(1.).set_color(color)
        return self

#------------------------------------- TEST -------------------------------------

#
# class Example(Scene):
#     def construct(self):
#         g = GVGraph(table=[[1,2,3,4,5],[6,7,8,9,10]])
#         g.hide_arrows().hide_numbers()
#         self.play(Create(g))
#         self.wait(2)

#         style_0 = {
#             'rows': [GRAY_A,GRAY_E], 
#             'cols': [RED,TEAL,TEAL,RED,RED], 
#             'nums': [BLACK,WHITE]
#         }
#         self.play(g.animate.set_style(**style_0))
#         self.wait(2)

#         self.play(*g.animate_arrows_end_with((0,1)))
#         self.play(g.animate.show_number((0,1)))
#         self.wait()
#         self.add(g.arrows.set_color(WHITE))
#         self.wait(2)

#         # self.play(g.animate.set_opacity(0.5))
#         # self.wait(2)

#         # self.play(pg.animate.test())


# class Example(Scene):
#     def construct(self):
#         hmm = HmmDiagram(
#             states=['F','B'],
#             symbols=['H','T'],
#             transition_prob=[[.9, .1],[.1, .9]],
#             emission_prob=[[.5,.5],[.75,.25]]
#         )
#         self.play(FadeIn(hmm))
#         self.wait()

#         hmm.get_state('F').set_fill_(GRAY_A).set_color_(BLACK).set_stroke_(GOLD)
#         hmm.get_state('B').set_fill_(GRAY_E).set_color_(WHITE).set_stroke_(GOLD)
#         hmm.get_symbol('H').set_stroke_(RED).set_fill_(BLACK)
#         hmm.get_symbol('T').set_stroke_(TEAL).set_fill_(BLACK)
#         self.play(FadeIn(hmm))
#         self.wait()

#         self.play(hmm.animate.hide_arrows_and_probs())
#         self.wait()

#------------------------------------- VIDEO -------------------------------------

class CoinExample(Scene):
    def construct(self):
        # show a Fair Coin and a Biased Coin
        coin_svg = SVGMobject('src/coin.svg', height=1.5)
        self.play(DrawBorderThenFill(coin_svg, run_time=2))
        self.wait()
        coin2_svg = coin_svg.copy()
        coins = VGroup(coin_svg, coin2_svg)
        self.remove(coin_svg)
        self.play(coins.animate(run_time=2).arrange(buff=1.5))
        self.wait()
        for t,c in zip([.7]*4, [YELLOW_D,WHITE]*2):
            self.play(coin_svg.animate.flip().set_fill(c, opacity=.5), run_time=t)
        self.wait()
        for t,c in zip([.3, 1.1]*2, [WHITE,YELLOW_D]*2):
            self.play(coin2_svg.animate.flip().set_fill(c, opacity=.5), run_time=t)
        self.wait()
        self.play(coins.animate(run_time=2).arrange(DOWN, buff=1.5))
        self.play(FadeOut(coins), run_time=1)

class WhatsHMM(Scene):
    def construct(self):
        hmm = HmmDiagram(
            states=['F','B'],
            symbols=['H','T'],
            transition_prob=[[.9, .1],[.1, .9]],
            emission_prob=[[.5,.5],[.25,.75]]
        )
        hmm.hide_arrows_and_probs()
        self.play(FadeIn(hmm))

        self.play(
            hmm.get_state('F').animate.set_fill_(WHITE).set_color_(BLACK).set_stroke_(WHITE),
            run_time = 1
        )
        self.play(
            hmm.get_state('B').animate.set_fill_(YELLOW_B).set_color_(BLACK).set_stroke_(YELLOW_B),
            run_time = 1
        )
        self.play(
            hmm.get_symbol('H').animate.set_stroke_(RED).set_fill_(BLACK),
            run_time = 1
        )
        self.play(
            hmm.get_symbol('T').animate.set_stroke_(BLUE).set_fill_(BLACK),
            run_time = 1
        )
        self.wait()

        self.play(GrowArrow(hmm.show_arrow('FH', color=[WHITE,RED])), GrowArrow(hmm.show_arrow('FT', color=[WHITE,BLUE])))
        self.wait()
        self.play(hmm.animate.show_prob('FH', color=RED))
        self.wait()
        self.play(hmm.animate.show_prob('FT', color=BLUE))
        self.wait()
        self.play(GrowArrow(hmm.show_arrow('BH', color=[YELLOW_B,RED])))
        self.wait()
        self.play(hmm.animate.show_prob('BH', color=RED))
        self.wait()
        self.play(GrowArrow(hmm.show_arrow('BT', color=[YELLOW_B,BLUE])))
        self.wait()
        self.play(hmm.animate.show_prob('BT', color=BLUE))
        self.wait()

        self.play(GrowFromEdge(hmm.show_arrow('FB', color=[WHITE,YELLOW_B]), edge=UP))
        self.wait()
        self.play(hmm.animate.show_prob('FB'))
        self.wait()
        self.play(GrowFromEdge(hmm.show_arrow('BF', color=[YELLOW_B,WHITE]), edge=DOWN))
        self.wait()
        self.play(hmm.animate.show_prob('BF'))
        self.wait()
        self.play(GrowFromCenter(hmm.show_arrow('FF', color=WHITE)))
        self.wait()
        self.play(hmm.animate.show_prob('FF'))
        self.wait()
        self.play(GrowFromCenter(hmm.show_arrow('BB', color=YELLOW_B)))
        self.wait()
        self.play(hmm.animate.show_prob('BB'))
        self.wait()

        transition = DecimalTable(
            table = [[.9,.1],[.1,.9]], 
            row_labels = [hmm.get_state('F').copy(),hmm.get_state('B').copy()],
            col_labels = [hmm.get_state('F').copy(), hmm.get_state('B').copy()],
            top_left_entry = Text('T', color=RED),
            v_buff=.5,
            h_buff=.5
        ).next_to(hmm, LEFT, buff=.8)
        emission = DecimalTable(
            table=[[.5,.5],[.25,.75]],
            element_to_mobject_config={'num_decimal_places': 2},
            row_labels = [hmm.get_state('F').copy(),hmm.get_state('B').copy()],
            col_labels = [hmm.get_symbol('H').copy(), hmm.get_symbol('T').copy()],
            top_left_entry = Text('E',color=RED),
            v_buff=.5,
            h_buff=.5
        ).next_to(hmm, RIGHT, buff=1.2)
        self.play(Write(transition, run_time=2))
        self.wait()
        self.play(Write(emission, run_time=2))
        self.wait()

class TheComplexity(Scene):
    def construct(self):
        background = FullScreenRectangle()
        background.set_fill(opacity=.5)
        background.set_color([BLUE,RED])
        self.play(GrowFromCenter(background, run_time=2))

        table = Table(
            table=[
                ['1','2','3','4','5','6'],
                ['T','H','T','H','H','H'],
                ['F','F','F','B','B','B'],
                ['0.5','0.9','0.9','0.1','0.9','0.9'],
                ['0.5','0.5','0.5','0.75','0.75','0.75']
            ],
            row_labels=[MathTex('i'),MathTex('x'),MathTex('\pi'),MathTex('transition(\pi_{i-1},\pi_{i})'),MathTex('emission_{\pi_i}(x_i)')],
            v_buff=0.5,
            h_buff=0.5,
            include_outer_lines=True,
        )
        rows = table.get_rows()
        
        self.play(Create(rows[1], run_time=2), lag_ratio=1)
        self.wait()
        self.play(Create(rows[0], run_time=2), lag_ratio=1)
        self.wait()
        self.play(Create(rows[2], run_time=2), lag_ratio=1)
        self.wait()

        self.play(Write(table.get_entries((4,2))))
        self.wait()
        self.play(Write(table.get_entries((4,1))))
        self.wait()
        self.play(Write(table.get_entries((5,2))))
        self.wait()
        self.play(Write(table.get_entries((5,1))))
        self.wait()
        self.play(Write(table.get_entries((4,3))))
        self.wait()
        self.play(Write(table.get_entries((5,3))))
        self.wait(.5)
        for c in [4,5,6,7]:
            for r in [4,5]:
                self.play(Write(table.get_entries((r,c))), run_time=1-c*.1)
        self.play(Write(table))
        self.wait()


        probs = VGroup(rows[3], rows[4]).copy()
        self.add(probs)
        self.play(table.animate.scale(0.3).to_corner(UL))
        self.wait()

        product_ = MathTex(r'Pr = \prod_{i=1}^{n}{T(\pi_{i-1},\pi_{i}) \times E_{\pi_i}(x_i) = 0.0017}')
        self.play(ReplacementTransform(probs, product_, run_time=2))
        self.wait()
        self.play(product_.animate(run_time=2).scale(.7).to_corner(UR))
        self.wait()

        chains = VGroup(rows[1], rows[2]).copy()
        self.play(chains.animate(run_time=2).scale(2).center())
        self.wait()

        bigO = MathTex('O(n)=|States|^n').next_to(chains, DOWN, buff=1)
        self.play(Write(bigO, run_time=2))
        self.wait()

        bad_svg = SVGMobject('src/bad.svg', height=1)
        bad_svg.set_color(RED).next_to(bigO, RIGHT)
        self.play(Create(bad_svg, run_time=2))

        self.wait()

class Viterbi(Scene):
    def construct(self):
        title = Text('Viterbi Graph').set_color([BLUE,RED])
        self.play(Write(title))
        self.play(title.animate(run_time=2).to_edge(UP))
        title_ul = Underline(title)
        self.play(Create(title_ul))
        self.wait()

        g = GVGraph(table=_V)
        g.hide_arrows().hide_numbers()

        symbols = VGroup()
        for symb, circ, color in zip('THTHHH', g.circles[0], style_0['cols']):
            symbols.add(
                GVNode(symb)\
                    .next_to(circ, UP)\
                    .set_color_(color)\
                    .set_fill_(BLACK)\
                    .set_stroke_(color)
            )

        states = VGroup()
        for stat, circ, bg_color, color in zip('FB', g.circles, style_0['rows'], style_0['nums']):
            states.add(
                GVNode(stat)\
                    .next_to(circ, LEFT)\
                    .set_color_(color)\
                    .set_fill_(bg_color)\
                    .set_stroke_(bg_color)
            )

        transition = DecimalTable(
            table = [[.9,.1],[.1,.9]], 
            row_labels = [states[0].copy(),states[1].copy()],
            col_labels = [states[0].copy(), states[1].copy()],
            top_left_entry = Text('T', color=RED),
            v_buff=.5,
            h_buff=.5,
        ).scale(.5).to_corner(UL)
        emission = DecimalTable(
            table=[[.5,.5],[.25,.75]],
            element_to_mobject_config={'num_decimal_places': 2},
            row_labels = [states[0].copy(),states[1].copy()],
            col_labels = [symbols[1].copy(), symbols[0].copy()],
            top_left_entry = Text('E',color=RED),
            v_buff=.5,
            h_buff=.5,
        ).scale(.5).next_to(transition, RIGHT)

        self.play(Create(symbols, run_time=3, lag_ratio=.5))
        self.wait()
        self.play(Create(states, lag_ratio=.5))
        self.wait(2)
        self.play(
            FadeIn(g.set_style(**style_0)), 
            symbols.animate.shift(DOWN*3).set_opacity(0),
            states.animate.shift(RIGHT*9).set_opacity(0),
            run_time=3,
        )
        self.wait()

        self.play(Write(transition), Write(emission), run_time=2)
        self.wait()

        pointer = Arrow().scale(.5).set_color(YELLOW).next_to(g.circles[0], LEFT)
        self.play(GrowFromEdge(pointer, LEFT), run_time=2)
        self.wait()

        init_prob = Text('0.5', color=YELLOW).scale(.4).next_to(pointer, UL)
        self.play(Write(init_prob))
        self.wait()
        e_FH, e_FT, e_BH, e_BT = emission.elements_without_labels.submobjects
        e_FT_ = e_FT.copy()
        self.play(Indicate(e_FT))
        self.play(e_FT_.animate.next_to(init_prob, RIGHT, buff=.5))
        self.wait()
        self.play(
            Flash(g.get_number((0,0)), flash_radius=.3), 
            g.animate.show_number((0,0))
        )

        self.play(
            pointer.animate.next_to(g.circles[1], LEFT),
            FadeOut(e_FT_)
        )
        self.play(init_prob.animate.next_to(pointer, UL))
        self.wait()
        self.play(Indicate(e_BT))
        e_BT_ = e_BT.copy()
        self.play(e_BT_.animate.next_to(init_prob, RIGHT, buff=.5))
        self.wait()
        self.play(
            Flash(g.get_number((1,0)), flash_radius=.3), 
            g.animate.show_number((1,0))
        )
        self.play(
            FadeOut(pointer),
            FadeOut(init_prob),
            FadeOut(e_BT_),
            run_time=2
        )

        self.play(*g.animate_arrows_end_with((0,1)))
        self.wait()

        arrow = Arrow().scale(.8).next_to(g, DL).shift(DOWN)
        arrow_txt = MathTex(r'= transition(\pi_{i-1},\pi_i) \times emission_{\pi_i}(x_i)').next_to(arrow, RIGHT)
        node = GVNode('pr').set_color_(WHITE).set_stroke_(WHITE).set_fill_(BLACK).next_to(arrow, DOWN)
        node_txt = MathTex(r'=Max(pr_{source} \times \qquad)').next_to(arrow_txt, DOWN, aligned_edge=LEFT)
        arrow_ = arrow.copy()
        self.play(Write(arrow))
        self.play(Write(arrow_txt))
        self.wait()
        self.play(Write(node))
        self.play(Write(node_txt), arrow_.animate.scale(.8).next_to(node_txt, RIGHT, buff=-1.1))
        self.wait()

        t_FF,t_FB,t_BF,t_BB = transition.elements_without_labels.submobjects
        t_FF_ = t_FF.copy()
        e_FH_ = e_FH.copy()
        self.play(t_FF_.animate.next_to(g.get_arrow(((0,0),(0,1))), UP,buff=0).shift(LEFT*.3))
        self.play(e_FH_.animate.next_to(t_FF_, RIGHT))
        self.wait()

        t_BF_ = t_BF.copy()
        e_FH__ = e_FH.copy()
        self.play(t_BF_.animate.next_to(g.get_arrow(((1,0),(0,1))), LEFT, buff=-0.4))
        self.play(e_FH__.animate.next_to(t_BF_, RIGHT))
        self.wait()

        self.play(
            Flash(g.get_number((0,1)), flash_radius=.3),
            g.animate.show_number((0,1)),
            FadeOut(t_BF_),
            FadeOut(e_FH__),
        )
        self.wait()

        self.play(
            *g.animate_arrows_end_with((1,1)),
            FadeOut(t_FF_),
            FadeOut(e_FH_),
            *[a.animate.set_color(WHITE) for a in g.get_arrows_end_with((0,1))],
            run_time=2
        )
        self.wait()
        self.play(
            g.animate.show_number((1,1)),
            Flash(g.get_number((1,1)), flash_radius=.3)
        )
        self.wait()

        last_pos = (1,1)
        for c in [2,3,4,5]:
            for r in [0,1]:
                cur_pos = (r,c)
                self.play(
                    *[a.animate.set_color(WHITE) for a in g.get_arrows_end_with(last_pos)],
                    *g.animate_arrows_end_with(cur_pos),
                    run_time=.8-c*.1
                )
                self.play(
                    g.animate.show_number(cur_pos),
                    Flash(g.get_number(cur_pos), flash_radius=.3),
                    run_time=.8-c*.1
                )
                last_pos = cur_pos
        self.play(*[a.animate.set_color(WHITE) for a in g.get_arrows_end_with(last_pos)])
        self.wait()

        for c in [5,4,3,2,1]:
            pos = ((0,c-1),(0,c))
            self.play(
                GrowFromEdge(
                    g.get_arrow(pos).set_color(YELLOW), 
                    RIGHT
                ), 
                run_time=1
            )
        self.wait()
        self.play(g.arrows.animate.set_color(GRAY_E))
        self.wait()

        self.play(
            FocusOn(g.get_circle((0,3))), 
            g.arrows.animate.set_color(GRAY_E),
            g.circles.animate.set_opacity(.4),
            g.numbers.animate.set_opacity(.4),
            run_time=3
        )
        self.add(g.get_circle((0,3)).set_opacity(1), g.get_number((0,3)).set_opacity(1))
        for c0 in range(2):
            for c1 in range(2):
                for c2 in range(2):
                    ss = [c0,c1,c2,0]
                    path = ss2path(ss)
                    self.play(*g.animate_path(path), run_time=2)

        self.wait()

class DecodingProblem(Scene):
    def construct(self):
        line0 = Line(LEFT*7, RIGHT*7).to_edge(UP)
        self.play(Write(line0))

        title = Text('Decoding Problem')\
            .set_color([BLUE,RED])\
            .next_to(line0, DOWN, aligned_edge=LEFT)
        self.play(Write(title, run_time=2))

        definition = Text('Find an optimal hidden path in an HMM given a string of its emitted symbols.', font_size=24)\
            .next_to(title, DOWN, aligned_edge=LEFT)
        line1 = line0.copy().next_to(definition, DOWN, aligned_edge=LEFT)
        self.play(Write(definition, run_time=2), GrowFromEdge(line1, RIGHT, run_time=3))
        self.wait()

        g = GVGraph(table=[[0.25, 0.1125, 0.0506, 0.0228, 0.0103, 0.0046], [0.125, 0.0281, 0.019, 0.0043, 0.001, 0.0003]])
        g.hide_numbers()
        self.play(FadeIn(g.set_style(**style_0), run_time=2))

        self.wait()

class Forward(Scene):
    def construct(self):
        title = Text('Forward Algorithm').set_color([BLUE,RED])
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        title_ul = Underline(title)
        self.play(Create(title_ul))
        self.wait()

        g = GVGraph(table=_F)
        g.hide_arrows().hide_numbers()

        symbols = VGroup()
        for symb, circ, color in zip('THTHHH', g.circles[0], style_0['cols']):
            symbols.add(
                GVNode(symb)\
                    .next_to(circ, UP)\
                    .set_color_(color)\
                    .set_fill_(BLACK)\
                    .set_stroke_(color)
            )

        states = VGroup()
        for stat, circ, bg_color, color in zip('FB', g.circles, style_0['rows'], style_0['nums']):
            states.add(
                GVNode(stat)\
                    .next_to(circ, LEFT)\
                    .set_color_(color)\
                    .set_fill_(bg_color)\
                    .set_stroke_(bg_color)
            )

        transition = DecimalTable(
            table = [[.9,.1],[.1,.9]], 
            row_labels = [states[0].copy(),states[1].copy()],
            col_labels = [states[0].copy(), states[1].copy()],
            top_left_entry = Text('T', color=RED),
            v_buff=.5,
            h_buff=.5,
        ).scale(.5).to_corner(UL)
        emission = DecimalTable(
            table=[[.5,.5],[.25,.75]],
            element_to_mobject_config={'num_decimal_places': 2},
            row_labels = [states[0].copy(),states[1].copy()],
            col_labels = [symbols[1].copy(), symbols[0].copy()],
            top_left_entry = Text('E',color=RED),
            v_buff=.5,
            h_buff=.5,
        ).scale(.5).next_to(transition, RIGHT)

        self.play(Create(symbols))
        self.wait()
        self.play(Create(states))
        self.wait()
        self.play(
            FadeIn(g.set_style(**style_0)), 
            symbols.animate.shift(DOWN*3).set_opacity(0),
            states.animate.shift(RIGHT*9).set_opacity(0),
            run_time=2,
        )

        self.play(Write(transition), Write(emission))
        self.play(g.animate.show_number((0,0)),run_time=.5)
        self.play(g.animate.show_number((1,0)),run_time=.5)
        self.wait()

        arrow = Arrow().scale(.8).next_to(g, DL).shift(DOWN)
        arrow_txt = MathTex(r'= transition(\pi_{i-1},\pi_i) \times emission_{\pi_i}(x_i)').next_to(arrow, RIGHT)
        node = GVNode('pr').set_color_(WHITE).set_stroke_(WHITE).set_fill_(BLACK).next_to(arrow, DOWN)
        node_txt = MathTex(r'=','Max',r'(pr_{source} \times \qquad)').next_to(arrow_txt, DOWN, aligned_edge=LEFT)
        sum_txt = SingleStringMathTex('Sum').set_color(YELLOW).move_to(node_txt[1])
        arrow_ = arrow.copy()
        self.play(Write(arrow))
        self.play(Write(arrow_txt))
        self.wait()
        self.play(Write(node))
        self.play(Write(node_txt), arrow_.animate.scale(.8).next_to(node_txt, RIGHT, buff=-1.1))
        self.wait()
        self.play(ReplacementTransform(node_txt[1], sum_txt))
        self.wait()

        last_pos = None
        for c in [1,2,3,4,5]:
            for r in [0,1]:
                cur_pos = (r,c)
                if last_pos:
                    self.play(
                        *[a.animate.set_color(WHITE) for a in g.get_arrows_end_with(last_pos)],
                        *g.animate_arrows_end_with(cur_pos),
                        run_time=.8-c*.1
                    )
                else:
                    self.play(
                        *g.animate_arrows_end_with(cur_pos),
                        run_time=.8-c*.1
                    )
                self.play(
                    g.animate.show_number(cur_pos),
                    Flash(g.get_number(cur_pos), flash_radius=.3),
                    run_time=.8-c*.1
                )
                last_pos = cur_pos
        self.play(*[a.animate.set_color(WHITE) for a in g.get_arrows_end_with(last_pos)])
        self.wait()

        self.play(
            FocusOn(g.get_circle((0,3))), 
            g.arrows.animate.set_color(GRAY_E),
            g.circles.animate.set_opacity(.4),
            g.numbers.animate.set_opacity(.4),
            run_time=3
        )
        self.add(g.get_circle((0,3)).set_opacity(1), g.get_number((0,3)).set_opacity(1))
        self.wait()

        gver = GVer().next_to(g)
        self.play(FadeIn(gver, target_position=ORIGIN, run_time=2))
        gver.wink(self, 2)
        self.wait()
        self.play(gver.animate(run_time=.5).watch(g.get_circle((0,3))))
        self.wait()

        for c0 in range(2):
            for c1 in range(2):
                for c2 in range(2):
                    for c3 in [0]:
                        path = ss2path([c0,c1,c2,c3])
                        self.play(*g.animate_path(path), run_time=2)
        gver.stop_watch()
        self.wait()

class Backward(Scene):
    def construct(self):
        title = Text('Backward Algorithm').set_color([BLUE,RED]).to_edge(UP).shift(RIGHT)
        title_ul = Underline(title)

        g = GVGraph(table=_B)
        g.hide_numbers()
        g.set_style(**style_0)
        g.arrows.set_color(GRAY_E)

        symbols = VGroup()
        for symb, circ, color in zip('THTHHH', g.circles[0], style_0['cols']):
            symbols.add(
                GVNode(symb)\
                    .next_to(circ, UP)\
                    .set_color_(color)\
                    .set_fill_(BLACK)\
                    .set_stroke_(color)
            )

        states = VGroup()
        for stat, circ, bg_color, color in zip('FB', g.circles, style_0['rows'], style_0['nums']):
            states.add(
                GVNode(stat)\
                    .next_to(circ, LEFT)\
                    .set_color_(color)\
                    .set_fill_(bg_color)\
                    .set_stroke_(bg_color)
            )

        transition = DecimalTable(
            table = [[.9,.1],[.1,.9]], 
            row_labels = [states[0].copy(),states[1].copy()],
            col_labels = [states[0].copy(), states[1].copy()],
            top_left_entry = Text('T', color=RED),
            v_buff=.5,
            h_buff=.5,
        ).scale(.5).to_corner(UL)
        emission = DecimalTable(
            table=[[.5,.5],[.25,.75]],
            element_to_mobject_config={'num_decimal_places': 2},
            row_labels = [states[0].copy(),states[1].copy()],
            col_labels = [symbols[1].copy(), symbols[0].copy()],
            top_left_entry = Text('E',color=RED),
            v_buff=.5,
            h_buff=.5,
        ).scale(.5).next_to(transition, RIGHT)

        self.add(title_ul, transition, emission)

        arrow = Arrow().scale(.8).next_to(g, DL).shift(DOWN)
        arrow_txt = MathTex(r'= transition(\pi_{i-1},\pi_i) \times emission_{\pi_i}(x_i)').next_to(arrow, RIGHT)
        node = GVNode('pr').set_color_(WHITE).set_stroke_(WHITE).set_fill_(BLACK).next_to(arrow, DOWN)
        node_txt = MathTex(r'=', r'Sum(', r'pr_{source}', r'\times \qquad)').next_to(arrow_txt, DOWN, aligned_edge=LEFT)
        node_txt[1].set_color(YELLOW)
        node_txt_ = SingleStringMathTex(r'pr_{target}')\
            .set_color(YELLOW)\
            .move_to(node_txt[2])
        arrow_ = arrow.copy().scale(.8).next_to(node_txt, RIGHT, buff=-1.1)
        self.add(arrow, arrow_txt, node, node_txt, arrow_)
        self.play(Write(title), FadeIn(g), run_time=2)
        self.wait()
        self.play(
            node_txt[1].animate.set_color(WHITE), 
            ReplacementTransform(node_txt[2], node_txt_),
            run_time=2
        )
        self.wait()

        self.play(g.animate.show_number((0,5)), run_time=1)
        self.play(g.animate.show_number((1,5)), run_time=1)
        self.wait()

        last_pos = None
        for c in [4,3,2,1,0]:
            for r in [0,1]:
                cur_pos = (r,c)
                if last_pos:
                    self.play(
                        *[a.animate.set_color(WHITE) for a in g.get_arrows_start_with(last_pos)],
                        *g.animate_arrows_start_with(cur_pos),
                        run_time=.2 + c*.1
                    )
                else:
                    self.play(
                        *g.animate_arrows_start_with(cur_pos),
                        run_time=.2 + c*.1
                    )
                self.play(
                    g.animate.show_number(cur_pos),
                    Flash(g.get_number(cur_pos), flash_radius=.3),
                    run_time=.2 + c*.1
                )
                last_pos = cur_pos
        self.play(*[a.animate.set_color(WHITE) for a in g.get_arrows_start_with(last_pos)], run_time=2)
        self.wait()

        self.play(
            FocusOn(g.get_circle((0,3))), 
            g.arrows.animate.set_color(GRAY_E),
            g.circles.animate.set_opacity(.4),
            g.numbers.animate.set_opacity(.4),
            run_time=2
        )
        self.add(g.get_circle((0,3)).set_opacity(1), g.get_number((0,3)).set_opacity(1))
        self.wait()
        for i in range(2):
            for j in range(2):
                ss = [i,j]
                path = ss2path([0]+list(ss), start_index=3)
                self.play(*g.animate_path(path), run_time=2)
        self.wait()

class ForBack(Scene):
    def construct(self):
        f_table = GVGraph(table=_F)\
            .set_style(**style_0)\
            .hide_arrows()
        b_table = GVGraph(table=_B)\
            .set_style(**style_0)\
            .hide_arrows()
        
        f_txt = Text('Forward ', color=BLUE).next_to(f_table, LEFT, buff=.4)
        b_txt = Text('Backward ', color=RED).next_to(b_table, LEFT, buff=.4)
        f_box = SurroundingRectangle(f_table, color=BLUE, corner_radius=.2, buff=.2)
        b_box = SurroundingRectangle(b_table, color=RED, corner_radius=.2, buff=.2)

        f = VGroup(f_table, f_txt, f_box).center().shift(UP*2)
        b = VGroup(b_table, b_txt, b_box).next_to(f,DOWN,aligned_edge=RIGHT).shift(DOWN)
        
        self.play(Write(f), run_time=3)
        self.play(Write(b), run_time=2)
        self.wait()

        point = (0,3)
        self.play(
            f_table.circles.animate.set_opacity(.3),
            b_table.circles.animate.set_opacity(.3),
        )
        self.play(
            f_table.get_circle(point).animate.set_opacity(1),
            b_table.get_circle(point).animate.set_opacity(1),
            run_time=.1
        )
        self.wait()

        gver = GVer().shift(LEFT*5)
        self.play(FadeIn(gver, run_time=2, target_position=ORIGIN))
        gver.wink(self, 2)
        self.wait()
        self.play(gver.animate(run_time=.5).watch(f_table))

        for c0 in range(2):
            for c1 in range(2):
                for c2 in range(2):
                    for c3 in [0]:
                        path = ss2path([c0,c1,c2,c3])
                        self.play(f_table.animate.set_path_color(path, BLUE), run_time=.1)
                        self.wait()
                        self.play(f_table.animate.hide_arrows(), run_time=.1)
        self.wait()

        self.play(gver.animate(run_time=.5).watch(b_table))
        gver.wink(self, 1)
        for c4 in range(2):
            for c5 in range(2):
                path = ss2path([c3,c4,c5], start_index=3)
                self.play(b_table.animate.set_path_color(path, RED), run_time=.1)
                self.wait()
                self.play(b_table.animate.hide_arrows(), run_time=.1)
        self.play(gver.animate.stop_watch())

        self.play(
            Circumscribe(f_table.get_circle(point), fade_out=True, run_time=3),
            Circumscribe(b_table.get_circle(point), fade_out=True, run_time=3),
        )
        self.wait()

        self.play(
            f_table.circles.animate.set_opacity(.3),
            f_table.numbers.animate.set_opacity(.3),
            b_table.circles.animate.set_opacity(.3),
            b_table.numbers.animate.set_opacity(.3),
            run_time=.5
        )

        arrow = ((0,3),(1,4))
        self.play(
            f_table.animate.set_path_color([arrow]),
            b_table.animate.set_path_color([arrow]),
            run_time=2
        )
        gver.wink(self,2)
        self.wait()
        self.play(gver.animate.shift(LEFT*3).rotate(TAU*1/4).scale(.3))

        self.play(
            f_table.get_circle(arrow[0]).animate.set_opacity(1),
            f_table.get_number(arrow[0]).animate.set_opacity(1),
            b_table.get_circle(arrow[1]).animate.set_opacity(1),
            b_table.get_number(arrow[1]).animate.set_opacity(1),
        )
        self.wait()
        self.play(Circumscribe(f_table.get_circle(arrow[0]), fade_out=True, run_time=1))
        self.play(
            f_table.get_arrow(arrow).animate(run_time=2).set_color(YELLOW),
            b_table.get_arrow(arrow).animate(run_time=2).set_color(YELLOW),
        )
        self.play(Circumscribe(b_table.get_circle(arrow[1]), fade_out=True, run_time=1))
        self.wait()

        self.play(
            f_table.circles.animate.set_opacity(1),
            f_table.numbers.animate.set_opacity(1),
            b_table.circles.animate.set_opacity(1),
            b_table.numbers.animate.set_opacity(1),
            run_time=3
        )
        
        self.wait()

class BaumWelch(Scene):
    def construct(self):
        title = Text('Baum-Welch Algorithm').set_color([BLUE,RED]).to_edge(UP)
        title_ul = Underline(title).set_color([BLUE,RED])

        feature0 = Text("🎉 It's  a special case of the Expectation Maximization (EM) algorithm", font_size=24)\
            .next_to(title_ul, DOWN, aligned_edge=LEFT).shift(LEFT*2)
        feature1 = Text('🎢 High level steps of the EM:', font_size=24)\
            .next_to(feature0, DOWN, aligned_edge=LEFT)
        step0 = Tex(r'1) Start with initial probability estimates [T,E]. Initially set equal probabilities or define them randomly.', font_size=30)\
            .next_to(feature1, DOWN, aligned_edge=LEFT).shift(RIGHT)
        step1 = Tex(r'2) Compute expectation of how often each transition/emission has been used. We will estimate latent variables', font_size=30)\
            .next_to(step0, DOWN, aligned_edge=LEFT)
        step1_ = MathTex(r'[\xi,\gamma].').next_to(step1, RIGHT).scale(.5).shift(LEFT*3.5+DOWN*.2)
        step2 = Tex(r'3) Re-estimate the probabilities [T,E] based on those estimates (latent variable).', font_size=30)\
            .next_to(step1, DOWN, aligned_edge=LEFT)
        step3 = Tex(r'4) Repeat until convergence.', font_size=30)\
            .next_to(step2, DOWN, aligned_edge=LEFT)
        feature2 = Text('🎭 Estimate for both T, E :', font_size=24)\
            .next_to(step3, DOWN, aligned_edge=LEFT).shift(LEFT)
        est_t = MathTex(r'T_{l,k} = \frac{num(T_{l,k})}{\sum_j^{all} num(T_{l,j})}', font_size=28)\
            .next_to(feature2, DOWN, aligned_edge=LEFT).shift(RIGHT)
        est_e = MathTex(r'E_k(b) = \frac{num(E_k(b))}{\sum_c^{all} num(E_k(c))}', font_size=28)\
            .next_to(est_t, buff=3)
        title_ = VGroup(title, title_ul)
        tg = VGroup(feature0, feature1, step0, step1, step1_, step2, step3, feature2, est_t, est_e)
        self.play(Write(title))
        self.play(Write(title_ul))
        self.wait()
        self.play(Write(feature0, run_time=2))
        self.wait()
        self.play(Write(feature1, run_time=2))
        self.wait()
        self.play(Write(step0, run_time=2))
        self.wait()
        self.play(Write(step1, run_time=2))
        self.play(Write(step1_, run_time=.2))
        self.wait()
        self.play(Write(step2, run_time=2))
        self.wait()
        self.play(Write(step3, run_time=2))
        self.wait()
        self.play(Write(feature2, run_time=2))
        self.wait()
        self.play(Write(est_t, run_time=2))
        self.wait()
        self.play(Write(est_e, run_time=2))
        self.wait()

        self.play(Unwrite(tg, run_time=3))
        self.play(title_.animate.shift(LEFT*3))

        g = GVGraph(table=_B)
        g.hide_numbers()
        g.set_style(**style_0)
        # g.arrows.set_color(GRAY_E)
        symbols = VGroup()
        for symb, circ, color in zip('THTHHH', g.circles[0], style_0['cols']):
            symbols.add(
                GVNode(symb)\
                    .next_to(circ, UP)\
                    .set_color_(color)\
                    .set_fill_(BLACK)\
                    .set_stroke_(color)
            )
        states = VGroup()
        for stat, circ, bg_color, color in zip('FB', g.circles, style_0['rows'], style_0['nums']):
            states.add(
                GVNode(stat)\
                    .next_to(circ, LEFT)\
                    .set_color_(color)\
                    .set_fill_(bg_color)\
                    .set_stroke_(bg_color)
            )
        g_ = VGroup(g, symbols, states)
        self.play(Create(g_, run_time=2))
        self.play(g_.animate.scale(.5).to_edge(UR))

        xi = MathTex(
            r'T_{l,k}^i',
            r'&=pr(\pi_i=l,\pi_{i+1}=k|x,\theta)\\',
            r'&=\frac{pr(\pi_i=l,\pi_{i+1}=k,x|\theta)}{pr(x|\theta)}\\',
            r'&=\xi_{l,k}(i)'
        )
        self.play(Write(xi[0]))
        self.play(Write(xi[1]))
        self.wait()
        self.play(Write(xi[2]))
        self.wait()
        self.play(Write(xi[3]))
        self.wait()
        self.play(xi.animate(run_time=2).scale(.6).next_to(title_, DOWN, aligned_edge=LEFT))
        self.wait()

        trans = MathTex(r'T_{l,k}=\frac{\sum_{i=1}^{N-1}\xi_{l,k}(i)}{\sum_{i=1}^{N-1}\sum_k^{all}\xi_{l,k}(i)}')
        self.play(Write(trans))
        self.wait()
        self.play(trans.animate.scale(.6).next_to(xi,DOWN,aligned_edge=LEFT))
        self.wait()

        gamma = MathTex(
            r'E_{k}^i(b)',
            r'&=pr(\pi_i=k|x,\theta)\\',
            r'&=\frac{pr(\pi_i=k,x|\theta)}{pr(x|\theta)}\\',
            r'&=\gamma_{k}^i(b) \quad if \quad x_i=b\quad otherwise\quad 0'
        ).shift(RIGHT)
        self.play(Write(gamma[0]))
        self.play(Write(gamma[1]))
        self.wait()
        self.play(Write(gamma[2]))
        self.wait()
        self.play(Write(gamma[3]))
        self.wait()
        self.play(gamma.animate(run_time=2).scale(.6).next_to(xi, buff=.5))
        self.wait()

        emit = MathTex(r'E_{k}(b)=\frac{\sum_{i=1}^{N-1}\gamma_{k}(b)}{\sum_{i=1}^{N-1}\sum_c^{all}\gamma_{k}(c)}')\
            .shift(RIGHT)
        self.play(Write(emit))
        self.wait()
        self.play(emit.animate.scale(.6).next_to(gamma,DOWN, aligned_edge=LEFT))
        self.wait()

        init_pr = MathTex(r'pr_{init}=\gamma^{i=1}').shift(DOWN)
        self.play(Write(init_pr))
        self.play(init_pr.animate.scale(.6).next_to(emit, buff=1.5))
        self.wait()

        tex_ = VGroup(init_pr, emit, gamma, trans, xi)
        self.play(FadeOut(title_), FadeOut(g_), FadeOut(tex_), run_time=2)

# class PyCode(Scene):
#     def construct(self):
#         code = ImageMobject('src/code.png').scale(1.2).to_edge(UP).shift(UP*1.2)
#         self.play(FadeIn(code))
#         self.play(code.animate.to_edge(DOWN), run_time=45, rate_func=rate_functions.linear)
#         self.play(FadeOut(code))

# class CGExample(Scene):
#     def construct(self):
#         pass

#----------------------------------- utilities -----------------------------------

def ss2path(state_series, start_index=0):
    path = []
    start = None
    for i, s in enumerate(state_series):
        if start:
            end = (s, i + start_index)
            path.append((start, end))
            start = end
        else:
            start = (s, i + start_index)
    return path

style_0 = {
    'rows': [WHITE,YELLOW_B], 
    'cols': [BLUE,RED,BLUE,RED,RED,RED], 
    'nums': [BLACK,BLACK]
}

_F = [[0.25      , 0.13125   , 0.06359375, 0.03216797, 0.015354  , 0.00714716],[0.375     , 0.090625  , 0.07101562, 0.01756836, 0.00475708, 0.00145419]]
_B = [[0.02219849, 0.04791211, 0.10142188, 0.220625  , 0.475     , 1.        ],[0.00813794, 0.02552148, 0.03029688, 0.085625  , 0.275     ,1.         ]]
_V = [[0.25      , 0.1125    , 0.050625  , 0.02278125, 0.01025156, 0.0046132 ],[0.375     , 0.084375  , 0.05695313, 0.01281445, 0.00288325, 0.00064873]]

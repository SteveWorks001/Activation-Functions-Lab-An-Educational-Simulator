# -*- coding: utf-8 -*-
"""Activation Functions Lab - an educational neural-network activation simulator.

Standalone Tkinter application built on the Python standard library only.
Author label: SteveWorks001.
"""

import math
import tkinter as tk
from tkinter import ttk

AUTHOR = "SteveWorks001"
APP_TITLE_RU = "Activation Functions Lab - симулятор функций активации"
APP_TITLE_EN = "Activation Functions Lab - activation function simulator"

# ---------------------------------------------------------------------------
# 1. Numerically stable primitives
# ---------------------------------------------------------------------------


def sigmoid(x):
    """1 / (1 + e^(-x)) without overflow for large |x|."""
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_(x):
    return math.tanh(x)


def tanh_d(x):
    t = math.tanh(x)
    return 1.0 - t * t


def relu(x):
    return x if x > 0.0 else 0.0


def relu_d(x):
    return 1.0 if x > 0.0 else 0.0


def leaky_relu(x, alpha):
    return x if x > 0.0 else alpha * x


def leaky_relu_d(x, alpha):
    return 1.0 if x > 0.0 else alpha


def binary_step(x):
    return 1.0 if x >= 0.0 else 0.0


def binary_step_d(x):
    return 0.0


def softplus(x, beta):
    """(1/beta) * ln(1 + e^(beta*x)), written so that it never overflows.

    log1p(exp(-|z|)) underflows harmlessly to 0, so huge |z| stay finite.
    """
    z = beta * x
    return (max(z, 0.0) + math.log1p(math.exp(-abs(z)))) / beta


def softplus_d(x, beta):
    return sigmoid(beta * x)


# ---------------------------------------------------------------------------
# 2. Small helpers shared by the math layer and the GUI
# ---------------------------------------------------------------------------


def fmt(v, digits=6):
    """Fixed-point rendering used in the calculation panel."""
    if v is None:
        return "n/a"
    if v != v or v in (float("inf"), float("-inf")):
        return "%.3e" % v
    a = abs(v)
    if a >= 100000.0:
        return "%.3e" % v
    if a >= 100.0:
        digits = min(digits, 3)
    elif a >= 1.0:
        digits = min(digits, 4)
    s = "%.*f" % (digits, v)
    if s in ("-0.0", "-0.00", "-0.000", "-0.0000", "-0.000000"):
        s = s[1:]
    return s


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def nice_step(span, target=6):
    """Readable tick distance for a given axis span."""
    if span <= 0.0:
        return 1.0
    raw = span / float(target)
    mag = 10.0 ** math.floor(math.log10(raw))
    best = mag
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        cand = mult * mag
        if cand >= raw:
            best = cand
            break
    else:
        best = 10.0 * mag
    return best


def ticks(lo, hi, step):
    out = []
    if step <= 0.0:
        return out
    k = math.ceil(lo / step)
    v = k * step
    while v <= hi + step * 1e-9:
        if v > lo - step * 1e-9:
            out.append(round(v, 10))
        v += step
    return out


def tick_label(v, step):
    if step >= 1.0:
        return "%d" % int(round(v))
    digits = max(1, min(4, int(math.ceil(-math.log10(step)))))
    return ("%%.%df" % digits) % v


# ---------------------------------------------------------------------------
# 3. Activation catalog
# ---------------------------------------------------------------------------
# Each entry is pure data plus callables. Text lives in TEXTS under
# fn.<key>.<field>, formulas are language-neutral ASCII math.

PARAMS = {
    "alpha": {
        "min": 0.01,
        "max": 0.50,
        "default": 0.10,
        "step": 0.01,
        "formula": "alpha",
    },
    "beta": {
        "min": 0.5,
        "max": 20.0,
        "default": 1.0,
        "step": 0.5,
        "formula": "beta",
    },
}


class Activation(object):
    def __init__(self, key, value, deriv, formula, dformula,
                 param=None, kink=None, saturating=False,
                 out_range="(-inf, +inf)"):
        self.key = key
        self.value = value
        self.deriv = deriv
        self.formula = formula
        self.dformula = dformula
        self.param = param            # key into PARAMS or None
        self.kink = kink              # ("x=0", note-text-key) or None
        self.saturating = saturating
        self.out_range = out_range

    # -- evaluation ---------------------------------------------------------
    def f(self, x, params):
        if self.param:
            return self.value(x, params[self.param])
        return self.value(x)

    def df(self, x, params):
        if self.param:
            return self.deriv(x, params[self.param])
        return self.deriv(x)

    def has_param(self):
        return self.param is not None

    def is_kink(self, x):
        return self.kink is not None and abs(x) < 1e-12


def _make_catalog():
    return [
        Activation(
            key="sigmoid",
            value=sigmoid,
            deriv=sigmoid_d,
            formula="f(x) = 1 / (1 + e^(-x))",
            dformula="f'(x) = f(x) * (1 - f(x))",
            saturating=True,
            out_range="(0, 1)",
        ),
        Activation(
            key="tanh",
            value=tanh_,
            deriv=tanh_d,
            formula="f(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
            dformula="f'(x) = 1 - f(x)^2",
            saturating=True,
            out_range="(-1, 1)",
        ),
        Activation(
            key="relu",
            value=relu,
            deriv=relu_d,
            formula="f(x) = max(0, x)",
            dformula="f'(x) = 1, x > 0;  0, x < 0",
            kink="undefined_zero",
            out_range="[0, +inf)",
        ),
        Activation(
            key="leaky_relu",
            value=leaky_relu,
            deriv=leaky_relu_d,
            formula="f(x) = x, x > 0;  alpha * x, x <= 0",
            dformula="f'(x) = 1, x > 0;  alpha, x < 0",
            param="alpha",
            kink="undefined_zero_leaky",
            out_range="(-inf, +inf)",
        ),
        Activation(
            key="binary_step",
            value=binary_step,
            deriv=binary_step_d,
            formula="f(x) = 1, x >= 0;  0, x < 0",
            dformula="f'(x) = 0, x != 0",
            kink="undefined_zero_step",
            out_range="{0, 1}",
        ),
        Activation(
            key="softplus",
            value=softplus,
            deriv=softplus_d,
            formula="f(x) = (1 / beta) * ln(1 + e^(beta * x))",
            dformula="f'(x) = sigmoid(beta * x)",
            param="beta",
            out_range="(0, +inf)",
        ),
    ]


ACTIVATIONS = _make_catalog()
BY_KEY = {}
for _a in ACTIVATIONS:
    BY_KEY[_a.key] = _a
del _a

ORDER = [a.key for a in ACTIVATIONS]

# ---------------------------------------------------------------------------
# 4. Curve sampling (pure data, so it can be verified outside the GUI)
# ---------------------------------------------------------------------------


def build_curve(act, params, x_lo, x_hi, use_deriv=False, samples=240):
    """Return (points, discontinuous). points = [(x, y), ...] left to right."""
    if x_hi <= x_lo:
        raise ValueError("empty x range")
    samples = int(samples)
    if samples < 2:
        samples = 2
    xs = []
    for i in range(samples + 1):
        xs.append(x_lo + (x_hi - x_lo) * i / float(samples))
    if act.kink is not None and x_lo < 0.0 < x_hi:
        # Force exact neighbours around the break so the jump is pixel-true.
        xs = sorted(set(xs) | {x_lo / 2.0, 0.0, x_hi / 2.0})
    pts = []
    for x in xs:
        if use_deriv and act.kink == "undefined_zero_step" and abs(x) < 1e-12:
            y = 0.0
        else:
            y = act.df(x, params) if use_deriv else act.f(x, params)
        pts.append((x, y))
    # Only the step function itself jumps; its derivative is a flat line.
    return pts, (act.key == "binary_step" and not use_deriv)


def split_segments(pts, y_lo, y_hi):
    """Break a polyline where the sampled jump is a real discontinuation."""
    span = y_hi - y_lo
    if span <= 0.0:
        return [pts]
    segs = [[]]
    for i, p in enumerate(pts):
        if i > 0:
            jump = abs(p[1] - pts[i - 1][1])
            if jump > 0.5 * span:
                segs.append([])
        segs[-1].append(p)
    return [s for s in segs if s]


def y_bounds(pts, include_zero=True):
    ys = [p[1] for p in pts]
    if not ys:
        return -1.0, 1.0
    lo = min(ys)
    hi = max(ys)
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if hi - lo < 1e-9:
        hi, lo = hi + 0.5, lo - 0.5
    pad = 0.07 * (hi - lo)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# 5. Step-by-step calculation
# ---------------------------------------------------------------------------
# Returns a list of (role, text). Roles: formula / substitution / result /
# dformula / dsubstitution / dresult / note. Labels for the roles are taken
# from TEXTS, the math itself is language-neutral.


def calc_block(act, x, params, lang):
    lines = []
    xs = fmt(x, 2)
    v = act.f(x, params)
    d = act.df(x, params)
    undefined = act.is_kink(x) and act.kink == "undefined_zero_step"
    kink_note = tr(TEXTS[lang], "calc.kink_" + act.kink) if act.is_kink(x) else None

    if act.key == "sigmoid":
        e = math.exp(-x)
        lines.append(("substitution", "f(%s) = 1 / (1 + e^(%s))" % (xs, fmt(-x, 4))))
        lines.append(("substitution", "= 1 / (1 + %s) = 1 / %s" % (fmt(e, 6), fmt(1.0 + e, 6))))
        lines.append(("result", "f(%s) = %s" % (xs, fmt(v))))
        lines.append(("dsubstitution", "f'(%s) = %s * (1 - %s)" % (xs, fmt(v), fmt(v))))
        lines.append(("dsubstitution", "= %s * %s" % (fmt(v), fmt(1.0 - v))))
        lines.append(("dresult", "f'(%s) = %s" % (xs, fmt(d))))
    elif act.key == "tanh":
        ep = math.exp(x)
        em = math.exp(-x)
        lines.append(("substitution", "f(%s) = (e^(%s) - e^(%s)) / (e^(%s) + e^(%s))"
                      % (xs, fmt(x, 4), fmt(-x, 4), fmt(x, 4), fmt(-x, 4))))
        lines.append(("substitution", "= (%s - %s) / (%s + %s)"
                      % (fmt(ep, 5), fmt(em, 5), fmt(ep, 5), fmt(em, 5))))
        lines.append(("result", "f(%s) = %s" % (xs, fmt(v))))
        lines.append(("dsubstitution", "f'(%s) = 1 - (%s)^2" % (xs, fmt(v))))
        lines.append(("dsubstitution", "= 1 - %s" % fmt(v * v)))
        lines.append(("dresult", "f'(%s) = %s" % (xs, fmt(d))))
    elif act.key == "relu":
        branch = tr(TEXTS[lang], "calc.branch_relu_%s" % _sign_region(x))
        lines.append(("substitution", "f(%s) = max(0, %s)" % (xs, xs)))
        lines.append(("note", branch))
        lines.append(("result", "f(%s) = %s" % (xs, fmt(v))))
        if kink_note:
            lines.append(("dsubstitution", "f'(x) = 0, x < 0"))
            lines.append(("dsubstitution", "f'(x) = 1, x > 0"))
        else:
            lines.append(("dsubstitution", "f'(%s) = %s" % (xs, fmt(d))))
        lines.append(("note", tr(TEXTS[lang], "calc.branch_relu_d_%s" % _sign_region(x))))
        lines.append(("dresult", "f'(%s) = %s" % (xs, "-" if kink_note else fmt(d))))
    elif act.key == "leaky_relu":
        a = params["alpha"]
        if x > 0.0:
            lines.append(("substitution", "f(%s) = x = %s" % (xs, xs)))
            lines.append(("dsubstitution", "f'(%s) = 1" % xs))
        else:
            lines.append(("substitution", "f(%s) = alpha * x = %s * %s"
                          % (xs, fmt(a, 2), xs)))
            if kink_note:
                lines.append(("dsubstitution", "f'(x) = %s, x < 0" % fmt(a, 2)))
                lines.append(("dsubstitution", "f'(x) = 1, x > 0"))
            else:
                lines.append(("dsubstitution", "f'(%s) = alpha = %s" % (xs, fmt(a, 2))))
        lines.append(("note", tr(TEXTS[lang], "calc.branch_leaky_%s" % _sign_region(x))))
        lines.append(("result", "f(%s) = %s" % (xs, fmt(v))))
        lines.append(("dresult", "f'(%s) = %s" % (xs, "-" if kink_note else fmt(d))))
    elif act.key == "binary_step":
        cond = "x >= 0" if x >= 0.0 else "x < 0"
        lines.append(("substitution", "f(%s) = 1, %s" % (xs, cond) if x >= 0.0
                      else "f(%s) = 0, %s" % (xs, cond)))
        lines.append(("result", "f(%s) = %s" % (xs, fmt(v))))
        if undefined:
            lines.append(("dsubstitution", "f'(0) -> " + tr(TEXTS[lang], "calc.dstep_zero")))
            lines.append(("dresult", "f'(0) = -"))
        else:
            lines.append(("dsubstitution", "f'(%s) = 0, x != 0" % xs))
            lines.append(("dresult", "f'(%s) = %s" % (xs, fmt(d))))
    elif act.key == "softplus":
        b = params["beta"]
        z = b * x
        lines.append(("substitution", "f(%s) = (1 / %s) * ln(1 + e^(%s * %s))"
                      % (xs, fmt(b, 2), fmt(b, 2), xs)))
        if z > 50.0:
            lines.append(("substitution", "z = beta * x = %s, e^z >> 1" % fmt(z, 4)))
            lines.append(("note", tr(TEXTS[lang], "calc.softplus_asymptotic")))
            lines.append(("substitution", "(1 / %s) * ln(1 + e^(%s)) ~= (1 / %s) * %s"
                          % (fmt(b, 2), fmt(z, 4), fmt(b, 2), fmt(z, 4))))
        elif z < -50.0:
            lines.append(("substitution", "z = beta * x = %s, e^z ~= 0, ln(1 + e^z) ~= e^z"
                          % fmt(z, 4)))
            lines.append(("note", tr(TEXTS[lang], "calc.softplus_tiny")))
        else:
            ez = math.exp(z)
            lines.append(("substitution", "= (1 / %s) * ln(1 + %s)" % (fmt(b, 2), fmt(ez, 6))))
            lines.append(("substitution", "= (1 / %s) * ln(%s)" % (fmt(b, 2), fmt(1.0 + ez, 6))))
        lines.append(("result", "f(%s) = %s" % (xs, fmt(v))))
        s = sigmoid(z)
        lines.append(("dsubstitution", "f'(%s) = sigmoid(%s * %s) = sigmoid(%s)"
                      % (xs, fmt(b, 2), xs, fmt(z, 4))))
        lines.append(("dsubstitution", "= 1 / (1 + e^(%s))" % fmt(-z, 4)))
        lines.append(("dresult", "f'(%s) = %s" % (xs, fmt(s))))
    if kink_note:
        lines.append(("kink", kink_note))
    return lines


def _sign_region(x):
    if x > 1e-12:
        return "pos"
    if x < -1e-12:
        return "neg"
    return "zero"


def meaning_lines(act, x, params, lang):
    """Explain what the numbers mean, derived from the numbers themselves."""
    v = act.f(x, params)
    d = act.df(x, params)
    out = []
    base = tr(TEXTS[lang], "fn.%s.meaning" % act.key)
    if base:
        out.append(base)
    a = abs(d)
    if act.is_kink(x):
        bucket = "kink"
    elif a < 1e-9:
        bucket = "zero"
    elif a < 0.02:
        bucket = "tiny"
    elif a < 0.15:
        bucket = "weak"
    elif a < 0.85:
        bucket = "medium"
    else:
        bucket = "strong"
    out.append(tr(TEXTS[lang], "meaning.%s" % bucket))
    if act.saturating:
        limits = {"sigmoid": (0.0, 1.0), "tanh": (-1.0, 1.0)}[act.key]
        near = min(abs(v - limits[0]), abs(v - limits[1]))
        if near < 0.02:
            out.append(tr(TEXTS[lang], "meaning.saturated"))
    if act.key == "leaky_relu":
        out.append(tr(TEXTS[lang], "meaning.leaky_%s" % _sign_region(x)))
    if act.key == "relu" and x <= 0.0:
        out.append(tr(TEXTS[lang], "meaning.relu_dead_branch"))
    if act.key == "softplus":
        if v == 0.0:
            out.append(tr(TEXTS[lang], "meaning.softplus_underflow"))
        else:
            out.append(tr(TEXTS[lang], "meaning.softplus_gap"))
    return out


# ---------------------------------------------------------------------------
# 6. Localization
# ---------------------------------------------------------------------------


def tr(tree, path):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def tr_list(tree, path):
    node = tr(tree, path)
    if node is None:
        return []
    if isinstance(node, list):
        return node
    return [node]


FN_KEYS = ("name", "tagline", "explanation", "deriv_behavior", "characteristic",
           "limitation", "meaning")
EXP_IDS = ("normal", "large_neg", "large_pos", "near_zero", "relu_neg",
           "relu_pos", "relu_vs_leaky", "sigmoid_sat", "tanh_sat")
HELP_IDS = ("what_is", "f_of_x", "derivative", "backprop", "sigmoid", "tanh",
            "relu", "leaky_relu", "binary_step", "softplus", "saturation",
            "dead_relu", "how_to")

TEXTS = {}

TEXTS["ru"] = {
    "ui": {
        "title": "Activation Functions Lab - лаборатория функций активации",
        "col_functions": "Функция активации",
        "col_input": "Вход x",
        "col_params": "Параметр функции",
        "col_view": "Диапазон графика",
        "col_presets": "Готовый эксперимент",
        "col_language": "Язык",
        "btn_reset_x": "Сброс x",
        "btn_reset_all": "Сбросить всё",
        "lang_switch": "EN",
        "lang_hint": "Переключение языка сохраняет состояние",
        "slider_x": "x",
        "range_hint": "ось x: от %s до %s",
        "no_params": "У выбранных функций параметров нет.",
        "param_alpha": "alpha - наклон при x < 0",
        "param_beta": "beta - крутизна сглаживания",
        "tab_calc": "Расчёт",
        "tab_compare": "Сравнение",
        "tab_phen": "Явления",
        "tab_help": "Справка",
        "status_ready": "Выберите функцию и значение x",
        "sizes_hint": "Проверено на 1300x780, 1366x768, 1520x940",
    },
    "calc": {
        "title": "Пошаговый расчёт",
        "formula": "Формула",
        "substitution": "Подстановка",
        "result": "Результат",
        "dformula": "Производная",
        "dsubstitution": "Подстановка в f'(x)",
        "dresult": "Значение f'(x)",
        "note": "Ветвь",
        "kink": "Точка излома",
        "meaning": "Что это значит",
        "kink_undefined_zero": ("В точности x = 0 производная не определена: слева "
                                 "наклон 0, справа 1. Мы показываем соглашение 0 - "
                                 "так поступают все фреймворки."),
        "kink_undefined_zero_step": ("В точности x = 0 производная не существует: "
                                     "функция разрывна. Часто для наглядности рисуют "
                                     "нуль, но настоящее значение - undefined."),
        "kink_undefined_zero_leaky": ("В точности x = 0 производная не определена: слева "
                                      "наклон alpha, справа 1. Обе ветви дают в нуле одно и то "
                                      "же значение, поэтому скачок есть только у наклона. На "
                                      "графике производной в нуле нарисован alpha - это "
                                      "соглашение отрисовки, а не существование производной."),
        "branch_relu_pos": "x > 0, поэтому max(0, x) = x",
        "branch_relu_neg": "x < 0, поэтому max(0, x) = 0",
        "branch_relu_zero": "x = 0, обе ветви дают 0",
        "branch_relu_d_pos": "x > 0, наклон линейной ветви равен 1",
        "branch_relu_d_neg": "x < 0, функция постоянна, наклон 0",
        "branch_relu_d_zero": "x = 0 - точка излома",
        "branch_leaky_pos": "x > 0: наклонной ветви нет, f(x) = x - как у ReLU",
        "branch_leaky_neg": "x < 0: работает ветвь alpha * x, она отрицательна и не равна нулю",
        "branch_leaky_zero": "x = 0: обе ветви сходятся в нуле, скачок только у наклона",
        "softplus_asymptotic": ("При beta*x > 50 слагаемое 1 ничтожно: ln(1 + e^z) "
                               "практически равно z"),
        "softplus_tiny": ("При beta*x < -50 экспонента мала: ln(1 + e^z) "
                          "практически равно e^z, то есть почти 0"),
        "dstep_zero": "производная не существует",
    },
    "meaning": {
        "zero": "f'(x) = 0: через эту точку градиент не проходит вообще.",
        "tiny": "f'(x) почти нулевая: обратный ход получит лишь сотые доли сигнала.",
        "weak": "f'(x) мала: градиент заметно ослабевает на каждом слое.",
        "medium": "f'(x) умеренная: градиент проходит с обычным ослаблением.",
        "strong": "f'(x) близка к 1: градиент проходит почти без потерь.",
        "kink": "Точка излома: производная здесь не определена, работает соглашение.",
        "saturated": "Выход прижат к асимптоте - это область насыщения.",
        "leaky_pos": "Положительная ветвь совпадает с ReLU: наклон 1.",
        "leaky_neg": "Отрицательная ветвь наклонена на alpha, градиент сохраняется.",
        "leaky_zero": "В нуле обе ветви сходятся, разрыва выхода нет.",
        "relu_dead_branch": "Отрицательная ветвь даёт нулевой выход и нулевой градиент.",
        "softplus_gap": "Выход и наклон строго положительны: в отличие от ReLU, здесь нет ветви с нулевым наклоном.",
        "softplus_underflow": "При таком beta*x число e^z меньше машинного нуля: значения округляются до 0, хотя математически они положительны.",
    },
    "graph": {
        "f_title": "Функция f(x)",
        "df_title": "Производная f'(x)",
        "axis_x": "x",
        "axis_y": "f(x)",
        "axis_y_d": "f'(x)",
        "point": "x = %s   f(x) = %s",
        "point_d": "x = %s   f'(x) = %s",
        "point_d_undef": "x = %s   f'(x) не определена",
        "out_of_view": "x вне диапазона графика",
        "flat_note": "Кривая совпадает с осью x",
    },
    "cmp": {
        "title": "Таблица сравнения при x = %s",
        "hint": "Значения f(x) и f'(x) пересчитываются для текущего x.",
        "h_function": "Функция",
        "h_fx": "f(x)",
        "h_dfx": "f'(x)",
        "h_range": "Обычный выход",
        "h_deriv": "Поведение f'(x)",
        "h_char": "Главная особенность",
        "h_limit": "Главное ограничение",
        "visual_title": "Все функции на одном графике",
        "show_f": "f(x)",
        "show_df": "f'(x)",
        "visual_hint": "Вертикальная линия - текущий x.",
    },
    "phen": {
        "title": "Ключевые явления",
        "hint": "Нажмите «Показать», чтобы переставить симулятор на нужную область.",
        "action": "Показать",
        "applied": "Явление: %s",
    },
    "exp": {
        "title": "Эксперименты",
        "hint": "Пресет выставляет функцию, x, параметры и диапазон графика.",
        "applied": "Эксперимент: %s",
    },
    "help": {
        "title": "Учебный гид",
        "pick": "Раздел",
        "index": "Всего разделов: %d",
        "next": "Дальше",
        "prev": "Назад",
    },
    "status": {
        "func_set": "Функция: %s",
        "exp_applied": "Эксперимент загружен: %s",
        "phen_applied": "Демонстрация загружена: %s",
        "lang_set": "Язык интерфейса: русский",
        "reset_all": "Состояние сброшено к значениям по умолчанию",
        "reset_x": "x возвращён к 1.00",
        "param_set": "%s = %s",
        "range_set": "Диапазон оси x: [%s, %s]",
        "invalid": "Неверное значение x: %s",
        "param_bad": "Неверное значение параметра %s: %s",
        "out_of_range": "x = %s вне диапазона графика [%s, %s]",
    },
    "val": {
        "empty": "пусто",
        "number": "нужно число",
        "too_big": "по модулю не больше 100",
        "param": "значение вне разрешённого диапазона",
    },
    "fn": {
        "sigmoid": {
            "name": "Сигмоида",
            "tagline": "Гладкая S-кривая от 0 до 1",
            "explanation": ("Сигмоида сжимает любое число в интервал (0, 1), поэтому "
                            "исторически использовалась как вероятность. Её производная "
                            "f' = f(1-f) максимальна ровно в нуле и равна 0.25, а уже "
                            "при |x| = 5 падает ниже 0.007. В насыщенной области выход "
                            "почти не меняется, значит и градиент почти исчезает: "
                            "предыдущие слои получают всё меньше информации."),
            "deriv_behavior": "Максимум 0.25 в нуле, убывает к 0 с ростом |x|",
            "characteristic": "Выход всегда между 0 и 1, кривая гладкая",
            "limitation": "Насыщение и затухающий градиент, выход не центрирован",
            "meaning": "Сигмоида отвечает на вопрос «насколько вход близок к положительному».",
        },
        "tanh": {
            "name": "Тангенс (tanh)",
            "tagline": "S-кривая от -1 до 1, центрированная",
            "explanation": ("Tanh - та же S-форма, но выход колеблется вокруг нуля и "
                            "лежит в (-1, 1). Поэтому производная 1 - f(x)^2 равна 1 в "
                            "нуле: около нуля tanh ведёт себя почти как тождественная "
                            "функция и градиент почти не теряется. При |x| больше 3 "
                            "функция прижимается к -1 или +1, и производная снова уходит "
                            "к нулю."),
            "deriv_behavior": "Равна 1 в нуле, убывает к 0 в насыщении",
            "characteristic": "Выход центрирован вокруг 0",
            "limitation": "Тоже насыщается: большие входы обрезаются в +-1",
            "meaning": "Tanh показывает знак и силу отклика вокруг нуля.",
        },
        "relu": {
            "name": "ReLU",
            "tagline": "Максимум из нуля и x",
            "explanation": ("ReLU оставляет положительные значения без изменений и "
                            "обнуляет отрицательные. Производная равна 1 там, где вход "
                            "положителен, поэтому градиент проходит без затухания - в "
                            "этом главная причина популярности функции. Там, где вход "
                            "отрицателен, производная строго нулевая: нейрон, который "
                            "всегда отрицателен, перестаёт учиться."),
            "deriv_behavior": "1 при x > 0, 0 при x < 0, излом в нуле",
            "characteristic": "Линейна на активной ветви, вычисления дешевле экспонент",
            "limitation": "Нулевой градиент на отрицательной ветви - риск «мёртвого» нейрона",
            "meaning": "ReLU либо пропускает сигнал целиком, либо выключает его.",
        },
        "leaky_relu": {
            "name": "Leaky ReLU",
            "tagline": "ReLU с небольшим наклоном слева",
            "explanation": ("Leaky ReLU оставляет положительную ветвь ReLU без изменений, "
                            "а отрицательную заменяет пологой линией с наклоном alpha. Выход "
                            "слева мал, но не нулевой, и производная там равна alpha, а не 0. "
                            "Именно это отличает её от ReLU: уходя в отрицательную область, "
                            "нейрон всё ещё получает градиент и может вернуться к жизни."),
            "deriv_behavior": "1 справа, alpha слева - разрыв в нуле",
            "characteristic": "Сохраняет малый градиент на отрицательных входах",
            "limitation": "alpha приходится выбирать вручную, кривая не гладкая в нуле",
            "meaning": "Leaky ReLU - это страховка от нулевого градиента ценой погрешности.",
        },
        "binary_step": {
            "name": "Пороговая функция",
            "tagline": "Жёсткое решение 0 или 1",
            "explanation": ("Пороговая функция выдаёт 1 при x >= 0 и 0 при x < 0. Она "
                            "реализует само понятие «нейрон сработал», но разрывна: в нуле "
                            "функция подпрыгивает с 0 до 1. Производная равна 0 всюду, кроме "
                            "нуля, где она не существует. Обратный ход по градиенту не может "
                            "улучшать такую функцию: малое изменение входа ничего не меняет, а "
                            "в скачке нет касательной."),
            "deriv_behavior": "0 при x != 0, не существует при x = 0",
            "characteristic": "Резкое бинарное решение, ровно две величины на выходе",
            "limitation": "Недифференцируемость: градиентное обучение невозможно",
            "meaning": "Пороговая функция показывает класс, но не показывает «насколько».",
        },
        "softplus": {
            "name": "Softplus",
            "tagline": "Мягкая версия ReLU",
            "explanation": ("Softplus - гладкое приближение ReLU: ln(1 + e^x). При больших "
                            "положительных входах она совпадает с x, при больших отрицательных "
                            "стремится к 0, но никогда не достигает нуля. Производная softplus "
                            "равна sigmoid(beta * x) - при beta = 1 это обычный sigmoid(x): "
                            "тот же сигмоидный переход, но теперь между 0 и 1, "
                            "а не между 0 и 0.25. Параметр beta управляет крутизной перехода: "
                            "чем больше beta, тем ближе кривая к ReLU."),
            "deriv_behavior": "Плавный рост от 0 до 1, разрывов нет",
            "characteristic": "Везде дифференцируема, выход строго положителен",
            "limitation": "Тяжелее ReLU по вычислениям и тоже не центрирована",
            "meaning": "Softplus показывает, как выглядит «сглаженный» излом ReLU.",
        },
    },
    "legend": {
        "sigmoid": "Сигмоида",
        "tanh": "Tanh",
        "relu": "ReLU",
        "leaky_relu": "Leaky ReLU",
        "binary_step": "Шаг",
        "softplus": "Softplus",
    },
    "exp_text": {},
    "help_text": {},
}

TEXTS["ru"]["exp_text"] = {
    "normal": ("Обычная область",
               "x = 1.0 у сигмоиды: вход ещё в рабочей зоне, кривая там крутая, "
               "поэтому f'(1) = 0.1966 - градиент живой. Сравните с экспериментом "
               "«большие положительные x»: та же функция, но производная почти исчезла."),
    "large_neg": ("Большой отрицательный x",
                  "x = -8: сигмоида прижалась к нижней асимптоте 0, f' = 0.000335. "
                  "Дальнейшее уменьшение входа почти не меняет выход, значит и градиент "
                  "по такому входу не течёт."),
    "large_pos": ("Большой положительный x",
                  "x = 8: выход 0.99966, то есть «почти единица», а производная та же "
                  "0.000335. Сигмоида насыщается симметрично с обеих сторон."),
    "near_zero": ("Около нуля",
                  "x = 0.1 у tanh: кривая здесь почти прямая, f'(0.1) = 0.99. Именно "
                  "в этой зоне производная максимальна, поэтому малые изменения входа "
                  "сильнее всего влияют на обучение."),
    "relu_neg": ("ReLU: отрицательная область",
                 "x = -3 даёт ровно 0, а f'(-3) = 0. Нейрон с таким входом не вносит "
                 "изменений в градиент - он «спит»."),
    "relu_pos": ("ReLU: положительная область",
                 "x = 3 проходит насквозь: f(3) = 3 и f'(3) = 1. Градиент не "
                 "ослабевает вообще - в этом сила ReLU."),
    "relu_vs_leaky": ("ReLU против Leaky ReLU",
                      "Один и тот же x = -2: у ReLU выход 0 и градиент 0, у Leaky ReLU при "
                      "alpha = 0.1 выход -0.2 и градиент 0.1. Разница маленькая по значению "
                      "и принципиальная по смыслу: второй нейрон продолжает учиться."),
    "sigmoid_sat": ("Насыщение сигмоиды",
                    "x = %s: f = %s, f' = %s. Сравните с f'(0) = 0.25: ослабление "
                    "в сто раз. В глубокой сети такие множители перемножаются по слоям."),
    "tanh_sat": ("Насыщение tanh",
                 "x = %s: f = %s, f' = %s. Гиперболический тангенс центрирован, "
                 "но от насыщения это его не спасает."),
}

TEXTS["ru"]["help_text"] = {
    "what_is": ("Что такое функция активации",
                ["Нейрон сначала считает взвешенную сумму входов z = w1*x1 + ... + wn*xn + b. "
                 "Функция активации применяет к этому z одно и то же правило для всех "
                 "нейронов слоя: a = f(z). Она решает, насколько сильно нейрон "
                 "«ответил» на вход.",
                 "Без функции активации слой - это просто матричное умножение. "
                 "Композиция линейных операций снова линейна: сколько бы слоёв ни "
                 "соединить, результат всегда сворачивается к одной линейной функции. "
                 "Такая сеть не способна нарисовать ни одну сложную границу классов.",
                 "Нелинейная f ломает эту замкнутость: каждый слой получает способность "
                 "гнуть пространство признаков, и сеть становится универсальным "
                 "приближателем. Отсюда вывод: активация нужна не для «биологичности», а "
                 "для выразительной силы модели."]),
    "f_of_x": ("Что такое f(x)",
               ["f(x) - это правило: входное число x превращается в выходное число f(x). "
                "В симуляторе левая колонка задаёт x, центр показывает f(x) как точку на "
                "кривой, справа - точное значение.",
                "Эти величины читаются вместе: f(x) отвечает на вопрос «какой сигнал пойдёт дальше», а "
                "производная f'(x) - на вопрос «насколько сильно этот сигнал зависит от "
                "входа». Одно и то же число x даёт обе величины, и их надо читать вместе."]),
    "derivative": ("Что такое производная",
                   ["Производная f'(x) - это наклон касательной к графику в точке x, то "
                    "есть скорость изменения выхода по входу: f'(x) = lim (f(x+h) - f(x)) / h.",
                    "f'(x) = 1 означает «изменили вход на 0.01 - выход изменился на 0.01». "
                    "f'(x) = 0.1 - «выход сдвинулся в десять раз слабее». f'(x) = 0 - «выход "
                    "не сдвинулся вовсе».",
                    "В точках излома (ReLU) и разрыва (пороговая функция) производная не "
                    "существует; на практике выбирают соглашение, и симулятор честно "
                    "помечает такие точки."]),
    "backprop": ("Зачем производная в обратном распространении",
                 ["Обучение уменьшает ошибку, сдвигая веса в противоположную сторону "
                  "градиента. Градиент по весу слоя - это цепочка множителей, и один из "
                  "множителей в ней всегда f'(z) соответствующего нейрона.",
                  "Если f'(z) мала, вклад нейрона в градиент мал: веса почти не двигаются, "
                  "даже когда ошибка большая. Если f'(z) = 0, вклад ровно нулевой - "
                  "обучение на этом нейроне останавливается.",
                  "Именно поэтому для выбора активации производная важнее, чем форма самой "
                  "кривой: вам нужен не красивый график, а устойчивый поток градиента."]),
    "sigmoid": ("Сигмоида",
                ["f(x) = 1 / (1 + e^(-x)), выход лежит в (0, 1), производная f' = f(1-f).",
                 "Максимум производной - 0.25 - приходится на x = 0. Уже при x = 3 она "
                 "равна 0.045, при x = 6 - 0.0025. Сеть из десяти слоёв с сигмоидами "
                 "умножит такие множители и получит градиент порядка 1e-6.",
                 "Второй недостаток: выход всегда положителен, поэтому все веса одного "
                 "слоя получают градиенты одного знака - оптимизация идёт «зигзагом». "
                 "Сигмоиду стоит выбирать, когда нужен именно вероятностный выход, а не "
                 "как универсальную активацию скрытого слоя."]),
    "tanh": ("Тангенс гиперболический",
             ["f(x) = (e^x - e^(-x)) / (e^x + e^(-x)), выход в (-1, 1), f' = 1 - f^2.",
              "Центрирование - главное преимущество перед сигмоидой: f'(0) = 1, а не 0.25, "
              "и знак отклика сохраняется. В окрестности нуля tanh ведёт себя почти как "
              "тождественная функция.",
              "Насыщение остаётся: при |x| >= 4 выход прижат к +-1, а производная близка к "
              "нулю. Tanh лучше сигмоиды для скрытых слоёв, но не решает проблему "
              "затухающего градиента полностью."]),
    "relu": ("ReLU",
             ["f(x) = max(0, x), производная 1 при x > 0 и 0 при x < 0.",
              "Положительная ветвь не сжимает градиент вообще: множитель равен 1. Поэтому "
              "ReLU сделала возможным обучение глубоких сетей и остаётся вариантом по "
              "умолчанию.",
              "Плата за это - потеря отрицательной ветви и излом в нуле. Зато вычисления "
              "не требуют экспонент, а сеть становится разреженной: активна только часть "
              "нейронов."]),
    "leaky_relu": ("Leaky ReLU",
                   ["f(x) = x при x > 0 и alpha*x при x <= 0; производная 1 и alpha "
                    "соответственно.",
                    "Проблема ReLU в том, что у нейрона, ушедшего в отрицательную область "
                    "по всем входам, градиент равен нулю и он не может вернуться. Leaky "
                    "ReLU даёт этому нейрону маленький, но ненулевой наклон - шанс "
                    "восстановиться сохраняется.",
                    "alpha выбирают небольшим (0.01 - 0.1): слишком большой наклон "
                    "искажает форму, слишком малый почти не помогает. В симуляторе можно "
                    "подвигать alpha и посмотреть, как меняется линия слева."]),
    "binary_step": ("Пороговая функция",
                    ["f(x) = 1 при x >= 0 и 0 при x < 0. Это исторический активатор "
                     "персептрона: он выдаёт класс, а не степень уверенности.",
                     "Производная равна нулю всюду, где она существует, а в нуле функция "
                     "разрывна и производной нет вовсе. Ни один малый сдвиг веса не "
                     "улучшает выход - градиентное обучение бесполезно.",
                     "Иногда рисуют «дельта-функцию» как производную скачка. Это "
                     "обобщённая функция, не число: использовать её в градиентном спуске "
                     "нельзя. Симулятор показывает производную равной нулю и явно помечает "
                     "точку x = 0."]),
    "softplus": ("Softplus",
                 ["f(x) = ln(1 + e^x) - гладкое приближение ReLU. При beta = 1 производная "
                  "softplus равна sigmoid(x), то есть плавно растёт от 0 до 1.",
                  "Разница с ReLU видна у нуля: ReLU имеет излом, а softplus переходит "
                  "плавно и остаётся строго положительной. Это удобно там, где нужна "
                  "дифференцируемость второго порядка.",
                  "Параметр beta меняет крутизну перехода: f(x) = (1/beta)*ln(1 + e^(beta*x)). "
                  "При большом beta softplus сходится к ReLU, при маленьком beta - к "
                  "почти прямой линии с малым наклоном. Beta не меняет смысл функции, а "
                  "показывает цену гладкости."]),
    "saturation": ("Насыщение и затухающий градиент",
                   ["Область, где кривая выходит на асимптоту, называется насыщением. В ней "
                    "f'(x) близка к нулю: вход можно менять сильно, а выход почти стоит на "
                    "месте.",
                    "При обратном распространении множители f'(z) перемножаются по всем "
                    "слоям. Несколько последовательных 0.01 превращаются в 1e-6, и "
                    "ранние слои перестают учиться. Это и есть затухающий градиент.",
                    "Симптом в симуляторе: двигайте x по сигмоиде вправо от 4 и смотрите на "
                    "нижний график - производная опустится к нулю раньше, чем можно "
                    "ожидать. Tanh ведёт себя так же, только с двух сторон от нуля."]),
    "dead_relu": ("«Мёртвый» ReLU",
                  ["Это не «ReLU не работает на отрицательных входах». На отрицательном "
                   "входе нейрон просто выдаёт 0 - это нормально.",
                   "Нейрон становится мёртвым, когда по всем обучающим примерам его "
                   "взвешенная сумма оказывается отрицательной. Тогда f'(z) = 0 для каждого "
                   "примера, градиент по его весам равен нулю, и градиентный спуск никогда "
                   "их не изменит: нейрон выпадает из обучения навсегда.",
                   "Обычная причина - слишком большой шаг обучения, который уводит веса "
                   "далеко в отрицательную область за один шаг. Помогает малый шаг, "
                   "правильная инициализация и функции с ненулевой отрицательной ветвью "
                   "(Leaky ReLU)."],),
    "how_to": ("Как пользоваться симулятором",
               ["1. Выберите функцию в левой колонке. 2. Задайте x: поле ввода, слайдер "
                "или кнопки +-0.1 и +-1. Десятичный разделитель - точка или запятая. "
                "3. Читайте центр: два графика, f(x) сверху, f'(x) снизу; точка на каждом "
                "графике - текущее значение.",
                "Вкладка «Расчёт» показывает подстановку числа в формулу шаг за шагом, а "
                "не только результат. «Сравнение» собирает все шесть функций в таблицу при "
                "текущем x и рисует их кривые вместе. «Явления» и «Эксперименты» "
                "переставляют симулятор так, чтобы эффект был виден прямо на графике.",
                "Диапазон оси x меняется кнопками +-2, +-5, +-10, +-20. Если x вышел за "
                "диапазон, симулятор сообщит об этом в статусной строке. Сброс возвращает "
                "сигмоиду, x = 1.0 и диапазон +-5."]),
}

TEXTS["en"] = {
    "ui": {
        "title": "Activation Functions Lab - activation function simulator",
        "col_functions": "Activation function",
        "col_input": "Input x",
        "col_params": "Function parameter",
        "col_view": "Graph range",
        "col_presets": "Ready-made experiment",
        "col_language": "Language",
        "btn_reset_x": "Reset x",
        "btn_reset_all": "Reset all",
        "lang_switch": "RU",
        "lang_hint": "Switching language keeps the current state",
        "slider_x": "x",
        "range_hint": "x axis: %s to %s",
        "no_params": "The selected function has no parameters.",
        "param_alpha": "alpha - slope for x < 0",
        "param_beta": "beta - steepness of the smoothing",
        "tab_calc": "Calculation",
        "tab_compare": "Comparison",
        "tab_phen": "Phenomena",
        "tab_help": "Help",
        "status_ready": "Pick a function and a value of x",
        "sizes_hint": "Tested at 1300x780, 1366x768, 1520x940",
    },
    "calc": {
        "title": "Step-by-step calculation",
        "formula": "Formula",
        "substitution": "Substitution",
        "result": "Result",
        "dformula": "Derivative",
        "dsubstitution": "Substitution in f'(x)",
        "dresult": "Value of f'(x)",
        "note": "Branch",
        "kink": "Kink point",
        "meaning": "What it means",
        "kink_undefined_zero": ("Exactly at x = 0 the derivative does not exist: the slope "
                                "is 0 on the left and 1 on the right. We show the convention "
                                "0, which is what every framework uses."),
        "kink_undefined_zero_step": ("Exactly at x = 0 the derivative does not exist: the "
                                     "function is discontinuous there. Textbooks often draw 0 "
                                     "for clarity, but the real answer is undefined."),
        "kink_undefined_zero_leaky": ("Exactly at x = 0 the derivative does not exist: the "
                                      "slope is alpha on the left and 1 on the right. Both "
                                      "branches give the same value in zero, so only the slope "
                                      "jumps. The derivative graph draws alpha at zero - that "
                                      "is a plotting convention, not a claim that the "
                                      "derivative exists."),
        "branch_relu_pos": "x > 0, so max(0, x) = x",
        "branch_relu_neg": "x < 0, so max(0, x) = 0",
        "branch_relu_zero": "x = 0, both branches give 0",
        "branch_relu_d_pos": "x > 0, the slope of the linear branch is 1",
        "branch_relu_d_neg": "x < 0, the function is constant, slope 0",
        "branch_relu_d_zero": "x = 0 is the kink point",
        "branch_leaky_pos": "x > 0: no tilted branch here, f(x) = x - exactly like ReLU",
        "branch_leaky_neg": "x < 0: the alpha * x branch applies, negative but never flat zero",
        "branch_leaky_zero": "x = 0: both branches meet at zero, only the slope jumps there",
        "softplus_asymptotic": ("For beta*x > 50 the added 1 is negligible: ln(1 + e^z) "
                                "is practically equal to z"),
        "softplus_tiny": ("For beta*x < -50 the exponent is tiny: ln(1 + e^z) is "
                          "practically equal to e^z, which is close to 0"),
        "dstep_zero": "the derivative does not exist",
    },
    "meaning": {
        "zero": "f'(x) = 0: no gradient passes through this point at all.",
        "tiny": "f'(x) is almost zero: backpropagation gets only hundredths of the signal.",
        "weak": "f'(x) is small: the gradient weakens noticeably at every layer.",
        "medium": "f'(x) is moderate: the gradient passes with ordinary damping.",
        "strong": "f'(x) is close to 1: the gradient passes almost without loss.",
        "kink": "Kink point: the derivative is undefined here, a convention applies.",
        "saturated": "The output is pressed against the asymptote - this is saturation.",
        "leaky_pos": "The positive branch equals ReLU: slope 1.",
        "leaky_neg": "The negative branch is tilted by alpha, so a gradient survives.",
        "leaky_zero": "At zero both branches meet, the output stays continuous.",
        "relu_dead_branch": "The negative branch gives zero output and zero gradient.",
        "softplus_gap": "Output and slope stay strictly positive: unlike ReLU there is no zero-slope branch.",
        "softplus_underflow": ("For this beta*x the number e^z is smaller than machine zero: "
                               "the values round to 0 although they are positive mathematically."),
    },
    "graph": {
        "f_title": "Function f(x)",
        "df_title": "Derivative f'(x)",
        "axis_x": "x",
        "axis_y": "f(x)",
        "axis_y_d": "f'(x)",
        "point": "x = %s   f(x) = %s",
        "point_d": "x = %s   f'(x) = %s",
        "point_d_undef": "x = %s   f'(x) undefined",
        "out_of_view": "x is outside the graph range",
        "flat_note": "The curve lies on the x axis",
    },
    "cmp": {
        "title": "Comparison table at x = %s",
        "hint": "f(x) and f'(x) are recomputed for the current x.",
        "h_function": "Function",
        "h_fx": "f(x)",
        "h_dfx": "f'(x)",
        "h_range": "Output range",
        "h_deriv": "Derivative behavior",
        "h_char": "Main characteristic",
        "h_limit": "Main limitation",
        "visual_title": "All functions on one graph",
        "show_f": "f(x)",
        "show_df": "f'(x)",
        "visual_hint": "The vertical line marks the current x.",
    },
    "phen": {
        "title": "Key phenomena",
        "hint": "Press Show to move the simulator onto the interesting region.",
        "action": "Show",
        "applied": "Phenomenon: %s",
    },
    "exp": {
        "title": "Experiments",
        "hint": "A preset sets the function, x, parameters and the graph range.",
        "applied": "Experiment loaded: %s",
    },
    "help": {
        "title": "Learning guide",
        "pick": "Section",
        "index": "Sections available: %d",
        "next": "Next",
        "prev": "Back",
    },
    "status": {
        "func_set": "Function: %s",
        "exp_applied": "Experiment loaded: %s",
        "phen_applied": "Demonstration loaded: %s",
        "lang_set": "Interface language: English",
        "reset_all": "State reset to the default values",
        "reset_x": "x returned to 1.00",
        "param_set": "%s = %s",
        "range_set": "x axis range: [%s, %s]",
        "invalid": "Invalid value of x: %s",
        "param_bad": "Invalid value of the parameter %s: %s",
        "out_of_range": "x = %s is outside the graph range [%s, %s]",
    },
    "val": {
        "empty": "empty input",
        "number": "a number is required",
        "too_big": "absolute value must not exceed 100",
        "param": "value outside the allowed range",
    },
    "fn": {
        "sigmoid": {
            "name": "Sigmoid",
            "tagline": "A smooth S-curve between 0 and 1",
            "explanation": ("Sigmoid squeezes any number into (0, 1), which is why it was "
                            "used as a probability. Its derivative f' = f(1-f) peaks at "
                            "exactly 0.25 in x = 0 and drops below 0.007 once |x| = 5. In "
                            "the saturated region the output barely moves, so the gradient "
                            "barely flows: earlier layers receive less and less information."),
            "deriv_behavior": "Peak 0.25 at zero, decays to 0 as |x| grows",
            "characteristic": "Output always between 0 and 1, curve is smooth",
            "limitation": "Saturation and vanishing gradient, output is not centered",
            "meaning": "Sigmoid answers how close the input is to being positive.",
        },
        "tanh": {
            "name": "Tanh",
            "tagline": "An S-curve from -1 to 1, centered",
            "explanation": ("Tanh has the same S-shape but swings around zero and stays in "
                            "(-1, 1). Its derivative 1 - f(x)^2 equals 1 at zero, so near "
                            "zero tanh behaves almost like the identity function and the "
                            "gradient is not lost. Beyond |x| = 3 the output presses against "
                            "-1 or +1 and the derivative again runs to zero."),
            "deriv_behavior": "Equals 1 at zero, decays to 0 in saturation",
            "characteristic": "Output is centered around 0",
            "limitation": "Also saturates: large inputs clip to +-1",
            "meaning": "Tanh shows both the sign and the strength of the response.",
        },
        "relu": {
            "name": "ReLU",
            "tagline": "The maximum of zero and x",
            "explanation": ("ReLU passes positive values unchanged and zeroes out negative "
                            "ones. The derivative is 1 wherever the input is positive, so "
                            "the gradient travels without decay - the main reason the "
                            "function became popular. Where the input is negative the "
                            "derivative is exactly zero: a neuron that is always negative "
                            "stops learning."),
            "deriv_behavior": "1 for x > 0, 0 for x < 0, a kink at zero",
            "characteristic": "Linear on the active branch, cheaper than exponentials",
            "limitation": "Zero gradient on the negative branch - risk of a dead neuron",
            "meaning": "ReLU either passes the signal whole or switches it off.",
        },
        "leaky_relu": {
            "name": "Leaky ReLU",
            "tagline": "ReLU with a small slope on the left",
            "explanation": ("Leaky ReLU keeps the positive branch of ReLU and replaces the "
                            "negative branch with a shallow line of slope alpha. The output "
                            "on the left is small but not zero, and the derivative there is "
                            "alpha instead of 0. That is the whole difference: a neuron that "
                            "falls into the negative region still receives a gradient."),
            "deriv_behavior": "1 on the right, alpha on the left - jump at zero",
            "characteristic": "Keeps a small gradient for negative inputs",
            "limitation": "alpha must be chosen by hand, curve is not smooth at zero",
            "meaning": "Leaky ReLU is insurance against a zero gradient, paid in accuracy.",
        },
        "binary_step": {
            "name": "Binary Step",
            "tagline": "A hard decision: 0 or 1",
            "explanation": ("The step function returns 1 for x >= 0 and 0 for x < 0. It "
                            "implements the very idea of a firing neuron, but it is "
                            "discontinuous: at zero it jumps from 0 to 1. The derivative is 0 "
                            "everywhere it exists, and at the jump it does not exist at all. "
                            "Gradient-based training cannot improve such a function: a small "
                            "change of the input changes nothing, and a jump has no tangent."),
            "deriv_behavior": "0 for x != 0, does not exist at x = 0",
            "characteristic": "Sharp binary decision, exactly two output values",
            "limitation": "Non-differentiable: gradient training is impossible",
            "meaning": "The step function shows the class, never how strong the margin is.",
        },
        "softplus": {
            "name": "Softplus",
            "tagline": "A smoothed version of ReLU",
            "explanation": ("Softplus is a smooth approximation of ReLU: ln(1 + e^x). For "
                            "large positive inputs it equals x, for large negative ones it "
                            "approaches 0 without ever reaching it. Its derivative is "
                            "sigmoid(beta * x), which at beta = 1 is the plain sigmoid(x): "
                            "the same S-shaped transition, but between 0 and 1 "
                            "instead of 0 and 0.25. The beta parameter controls the "
                            "steepness: the larger beta, the closer the curve gets to ReLU."),
            "deriv_behavior": "Smooth growth from 0 to 1, no breaks",
            "characteristic": "Differentiable everywhere, output strictly positive",
            "limitation": "Costlier than ReLU and also not centered",
            "meaning": "Softplus shows what a rounded ReLU kink looks like.",
        },
    },
    "legend": {
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "relu": "ReLU",
        "leaky_relu": "Leaky",
        "binary_step": "Step",
        "softplus": "Softplus",
    },
    "exp_text": {
        "normal": ("Ordinary region",
                   "x = 1.0 on the sigmoid: the input is still inside the working zone, the "
                   "curve is steep there, so f'(1) = 0.1966 - a live gradient. Compare with "
                   "the large positive x experiment: same function, but the gradient has "
                   "almost vanished."),
        "large_neg": ("Large negative x",
                      "x = -8: the sigmoid presses against its lower asymptote 0, "
                      "f' = 0.000335. Making the input even smaller barely changes the "
                      "output, so no gradient flows along such an input."),
        "large_pos": ("Large positive x",
                      "x = 8: the output is 0.99966, almost one, and the derivative is the "
                      "same 0.000335. Sigmoid saturates symmetrically on both sides."),
        "near_zero": ("Near zero",
                      "x = 0.1 on tanh: the curve is almost straight here, f'(0.1) = 0.99. "
                      "The derivative is at its maximum in this zone, so small changes of "
                      "the input influence training the most."),
        "relu_neg": ("ReLU: negative region",
                    "x = -3 gives exactly 0 and f'(-3) = 0. A neuron with such an input "
                    "contributes nothing to the gradient - it is asleep."),
        "relu_pos": ("ReLU: positive region",
                     "x = 3 passes straight through: f(3) = 3 and f'(3) = 1. The gradient "
                     "does not weaken at all - that is the strength of ReLU."),
        "relu_vs_leaky": ("ReLU versus Leaky ReLU",
                          "The same x = -2: ReLU gives output 0 and gradient 0, Leaky ReLU "
                          "with alpha = 0.1 gives -0.2 and 0.1. The difference is small in "
                          "value and fundamental in meaning: the second neuron keeps "
                          "learning."),
        "sigmoid_sat": ("Sigmoid saturation",
                        "x = %s: f = %s, f' = %s. Compare with f'(0) = 0.25 - a "
                        "hundredfold weakening. In a deep network such factors multiply "
                        "across layers."),
        "tanh_sat": ("Tanh saturation",
                     "x = %s: f = %s, f' = %s. Tanh is centered, but that does "
                     "not save it from saturation."),
    },
    "help_text": {
        "what_is": ("What an activation function is",
                    ["A neuron first computes a weighted sum of its inputs, "
                     "z = w1*x1 + ... + wn*xn + b. The activation function applies one and the "
                     "same rule to every neuron of the layer: a = f(z). It decides how "
                     "strongly the neuron responded to the input.",
                     "Without an activation function a layer is just a matrix product. A "
                     "composition of linear operations is linear again: however many layers "
                     "you stack, the result collapses to one linear function. Such a network "
                     "cannot draw a single curved decision boundary.",
                     "A nonlinear f breaks that closure: every layer gains the ability to "
                     "bend the feature space, and the network becomes a universal "
                     "approximator. So activation is not about biology - it is about the "
                     "expressive power of the model."]),
        "f_of_x": ("What f(x) is",
                   ["f(x) is a rule: the input number x becomes the output number f(x). In "
                    "the simulator the left column sets x, the center shows f(x) as a point "
                    "on the curve, and the right panel gives the exact value.",
                    "The pair goes together: f(x) answers which signal travels onward, while "
                    "f'(x) answers how strongly that signal depends on the input. One number "
                    "x produces both quantities, and they must be read side by side."]),
        "derivative": ("What a derivative is",
                       ["The derivative f'(x) is the slope of the tangent to the graph at x, "
                        "that is the rate at which the output changes with the input: "
                        "f'(x) = lim (f(x+h) - f(x)) / h.",
                        "f'(x) = 1 means: move the input by 0.01 and the output moves by "
                        "0.01. f'(x) = 0.1 means the output moves ten times less. f'(x) = 0 "
                        "means the output does not move at all.",
                        "At kinks (ReLU) and jumps (the step function) the derivative does "
                        "not exist; practice picks a convention, and the simulator marks "
                        "such points explicitly."]),
        "backprop": ("Why derivatives matter in backpropagation",
                     ["Training reduces the error by moving weights against the gradient. The "
                      "gradient of a layer weight is a chain of factors, and one factor in "
                      "that chain is always f'(z) of the corresponding neuron.",
                      "If f'(z) is small, the neuron contributes little to the gradient: the "
                      "weights crawl even when the error is large. If f'(z) = 0, the "
                      "contribution is exactly zero and training of that neuron stops.",
                      "This is why the derivative, not the shape of the curve, should drive "
                      "the choice of activation: you need a steady flow of gradient, not a "
                      "pretty graph."]),
        "sigmoid": ("Sigmoid",
                    ["f(x) = 1 / (1 + e^(-x)), output in (0, 1), derivative f' = f(1-f).",
                     "The peak of the derivative, 0.25, sits at x = 0. At x = 3 it is "
                     "already 0.045, at x = 6 it is 0.0025. A network of ten sigmoid layers "
                     "multiplies such factors and ends up with a gradient near 1e-6.",
                     "The second drawback: the output is always positive, so all weights of a "
                     "layer receive gradients of one sign and optimization zigzags. Choose "
                     "sigmoid when you need a probability, not as a default hidden-layer "
                     "activation."]),
        "tanh": ("Tanh",
                 ["f(x) = (e^x - e^(-x)) / (e^x + e^(-x)), output in (-1, 1), "
                  "f' = 1 - f^2.",
                  "Centering is the main advantage over sigmoid: f'(0) = 1 rather than 0.25, "
                  "and the sign of the response survives. Near zero tanh acts almost like the "
                  "identity function.",
                  "Saturation remains: at |x| >= 4 the output sticks to +-1 and the "
                  "derivative is close to zero. Tanh beats sigmoid in hidden layers, but it "
                  "does not remove the vanishing gradient."]),
        "relu": ("ReLU",
                 ["f(x) = max(0, x), derivative 1 for x > 0 and 0 for x < 0.",
                  "The positive branch does not compress the gradient at all: the factor is "
                  "1. That is what made deep training practical, and ReLU is still the "
                  "default choice.",
                  "The price is the lost negative branch and the kink at zero. In exchange "
                  "the computation needs no exponentials, and the network becomes sparse: "
                  "only part of the neurons are active."]),
        "leaky_relu": ("Leaky ReLU",
                       ["f(x) = x for x > 0 and alpha*x for x <= 0; the derivative is 1 and "
                        "alpha respectively.",
                        "The trouble with ReLU is that a neuron whose weighted sum is "
                        "negative for every training example has f'(z) = 0 everywhere, so its "
                        "weight gradients are zero and gradient descent can never move them. "
                        "Leaky ReLU gives that neuron a small but nonzero slope, so it can "
                        "come back.",
                        "alpha is kept small (0.01 to 0.1): too large distorts the shape, too "
                        "small barely helps. In the simulator you can drag alpha and watch "
                        "the left branch tilt."]),
        "binary_step": ("Binary Step",
                        ["f(x) = 1 for x >= 0 and 0 for x < 0. This is the historical "
                         "activation of the perceptron: it returns a class, not a degree of "
                         "confidence.",
                         "The derivative is zero wherever it exists, and at zero the function "
                         "jumps, so there is no derivative at all. No small change of a "
                         "weight improves the output, which makes gradient training useless.",
                         "Textbooks sometimes draw a delta function as the derivative of the "
                         "jump. That is a generalized function, not a number, and it cannot "
                         "be used in gradient descent. The simulator shows the derivative as "
                         "zero and marks x = 0 as undefined."]),
        "softplus": ("Softplus",
                     ["f(x) = ln(1 + e^x), a smooth approximation of ReLU. At beta = 1 the "
                      "derivative of softplus equals sigmoid(x), rising smoothly from 0 to 1.",
                      "The difference from ReLU is visible near zero: ReLU has a kink, "
                      "softplus rounds the transition and stays strictly positive. This is "
                      "handy wherever a second derivative is needed.",
                      "The beta parameter changes the steepness: f(x) = (1/beta)*ln(1 + "
                      "e^(beta*x)). As beta grows, softplus converges to ReLU; as beta "
                      "shrinks, it becomes an almost straight shallow line. Beta does not "
                      "change the meaning of the function - it shows the price of smoothness."]),
        "saturation": ("Saturation and the vanishing gradient",
                       ["The region where the curve settles onto its asymptote is called "
                        "saturation. There f'(x) is close to zero: you can move the input a "
                        "lot while the output stays put.",
                        "During backpropagation the f'(z) factors multiply across all layers. "
                        "A few consecutive 0.01 turn into 1e-6, and the early layers stop "
                        "learning. That is the vanishing gradient.",
                        "The symptom in the simulator: drag x along the sigmoid past 4 and "
                        "watch the lower graph - the derivative reaches zero earlier than you "
                        "expect. Tanh does the same on both sides of zero."]),
        "dead_relu": ("The dead ReLU problem",
                      ["This is not \"ReLU fails on negative inputs\". On a negative input "
                       "the neuron simply outputs 0, which is normal.",
                       "A neuron dies when its weighted sum is negative for every training "
                       "example. Then f'(z) = 0 for all of them, the gradient of its weights "
                       "is zero, and gradient descent will never change them: the neuron "
                       "leaves training for good.",
                       "The usual cause is a learning rate too large, which pushes the "
                       "weights deep into the negative region in a single step. A smaller "
                       "step, sensible initialization and activations with a nonzero "
                       "negative branch (Leaky ReLU) help."]),
        "how_to": ("How to use this simulator",
                   ["1. Pick a function in the left column. 2. Set x: the input field, the "
                    "slider, or the +-0.1 and +-1 buttons. The decimal separator may be a "
                    "dot or a comma. 3. Read the center: two graphs, f(x) on top, f'(x) "
                    "below; the marker on each graph is the current value.",
                    "The Calculation tab shows the number substituted into the formula step "
                    "by step, not just the result. Comparison collects all six functions "
                    "into a table at the current x and draws their curves together. "
                    "Phenomena and Experiments move the simulator so that the effect is "
                    "visible on the graph right away.",
                    "The x axis range is set with the +-2, +-5, +-10, +-20 buttons. If x "
                    "falls outside the range the status line says so. Reset returns to "
                    "sigmoid, x = 1.0 and the +-5 range."]),
    },
}


def compare_text(key, field, lang):
    return tr(TEXTS[lang], "fn.%s.%s" % (key, field))


def exp_text(key, lang):
    node = tr(TEXTS[lang], "exp_text.%s" % key)
    if not node:
        return ("", "")
    body = node[1]
    if "%s" in body:
        # A caption that quotes numbers takes them from the very functions the
        # panels and the graphs use, so it cannot drift away from the math.
        st = EXPERIMENTS_BY_KEY[key]
        act = BY_KEY[st["fn"]]
        params = {"alpha": st.get("alpha", PARAMS["alpha"]["default"]),
                  "beta": st.get("beta", PARAMS["beta"]["default"])}
        x = st["x"]
        body = body % (fmt(x, 2), fmt(act.f(x, params)), fmt(act.df(x, params)))
    return node[0], body


def help_text(key, lang):
    node = tr(TEXTS[lang], "help_text.%s" % key)
    if not node:
        return ("", [])
    return node[0], node[1]


# ---------------------------------------------------------------------------
# 7. Theme
# ---------------------------------------------------------------------------

BG = "#f2f1ec"
PANEL = "#ffffff"
BORDER = "#d5d3cb"
INK = "#232323"
MUTED = "#6d6b64"
FAINT = "#7b7972"
GRID = "#e8e6df"
AXIS = "#a8a59c"
F_COLOR = "#1d6fa5"
D_COLOR = "#cf6a1c"
SEL_BG = "#1d6fa5"
SEL_FG = "#ffffff"
BTN_BG = "#fbfaf7"
BTN_ACTIVE = "#ecebe4"
ERR_BG = "#f7dcd6"
OK_BG = PANEL
LEGEND_COLORS = ["#1d6fa5", "#2f8f5b", "#cf6a1c", "#8b3fa8", "#b03040", "#6b6b2f"]

FONT = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_SECTION = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 12, "bold")
FONT_TAG = ("Segoe UI", 9, "italic")
MONO = ("Consolas", 10)
MONO_B = ("Consolas", 10, "bold")
MONO_SMALL = ("Consolas", 9)

X_RANGES = (2.0, 5.0, 10.0, 20.0)
MAX_ABS_X = 100.0


# ---------------------------------------------------------------------------
# 8. Custom Canvas plot widget (no matplotlib, no external libraries)
# ---------------------------------------------------------------------------


class PlotCanvas(tk.Canvas):
    """One framed graph: axes, grid, zero lines, curve(s), current marker."""

    def __init__(self, master, accent, name="", **kw):
        tk.Canvas.__init__(self, master, bg=PANEL, highlightthickness=1,
                           highlightbackground=BORDER, highlightcolor=BORDER,
                           bd=0, **kw)
        self.accent = accent
        self.name = name
        self.domain = (-5.0, 5.0)
        self.series = []
        self.marker = None
        self.note = ""
        self.legend = False
        self.title = ""
        self.subtitle = ""
        self.bind("<Configure>", lambda event: self.redraw())

    # -- data ---------------------------------------------------------------
    def set_series(self, series, domain, marker=None, note="", legend=False,
                   title="", subtitle="", xlabel="", ylabel=""):
        """series: list of dicts with keys pts, color, label, dashed, jumps."""
        self.series = series
        self.domain = domain
        self.marker = marker
        self.note = note
        self.legend = legend
        self.title = title
        self.subtitle = subtitle
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.redraw()

    def set_domain(self, domain):
        self.domain = domain
        self.redraw()

    # -- drawing ------------------------------------------------------------
    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 80 or h < 80:
            return
        ml, mr, mt, mb = 54, 26, 32, 24
        x0, y0, x1, y1 = ml, mt, w - mr, h - mb
        if x1 - x0 < 50 or y1 - y0 < 40:
            return
        x_lo, x_hi = self.domain
        if x_hi <= x_lo:
            return
        all_pts = []
        for s in self.series:
            all_pts.extend(s["pts"])
        if not all_pts:
            return
        y_lo, y_hi = y_bounds(all_pts)

        def X(v):
            return x0 + (v - x_lo) * (x1 - x0) / (x_hi - x_lo)

        def Y(v):
            return y1 - (v - y_lo) * (y1 - y0) / (y_hi - y_lo)

        # header
        if self.title:
            self.create_text(ml - 42, 6, anchor="w", text=self.title,
                             font=FONT_BOLD, fill=INK)
        if self.subtitle:
            self.create_text(w - mr, 6, anchor="e", text=self.subtitle,
                             font=MONO_SMALL, fill=self.accent)
        self.create_rectangle(x0 - 1, mt - 6, x1 + 1, y1 + 1, fill=PANEL,
                              outline=PANEL)

        # grid + tick labels
        x_step = nice_step(x_hi - x_lo, target=max(3, int((x1 - x0) / 90)))
        y_step = nice_step(y_hi - y_lo, target=max(3, int((y1 - y0) / 46)))
        for xv in ticks(x_lo, x_hi, x_step):
            px = X(xv)
            self.create_line(px, y0, px, y1, fill=GRID)
            self.create_text(px, y1 + 10, text=tick_label(xv, x_step),
                             font=FONT_SMALL, fill=MUTED)
        for yv in ticks(y_lo, y_hi, y_step):
            py = Y(yv)
            self.create_line(x0, py, x1, py, fill=GRID)
            self.create_text(x0 - 6, py, anchor="e",
                             text=tick_label(yv, y_step), font=FONT_SMALL,
                             fill=MUTED)

        # zero lines
        if x_lo < 0.0 < x_hi:
            self.create_line(X(0.0), y0, X(0.0), y1, fill=AXIS, width=2)
        if y_lo < 0.0 < y_hi:
            self.create_line(x0, Y(0.0), x1, Y(0.0), fill=AXIS, width=2)

        # axis names. The x name goes into the right margin: the last tick
        # label is centred on the axis end, so inside the plot the two collide.
        self.create_text(x1 + 8, y1 + 10, anchor="w",
                         text=getattr(self, "xlabel", ""), font=FONT_SMALL,
                         fill=FAINT)
        if getattr(self, "ylabel", ""):
            self.create_text(11, (y0 + y1) / 2.0, anchor="w", angle=90,
                             text=self.ylabel, font=FONT_SMALL, fill=FAINT)

        # curves
        for s in self.series:
            color = s.get("color", self.accent)
            dashed = s.get("dashed", False)
            segs = split_segments(s["pts"], y_lo, y_hi)
            for seg in segs:
                flat = []
                for (px_, py_) in [(X(p[0]), Y(p[1])) for p in seg]:
                    flat.extend([px_, py_])
                if dashed:
                    self.create_line(flat, fill=color, width=2, dash=(5, 3))
                else:
                    self.create_line(flat, fill=color, width=2.4 if not self.legend else 2)
            if len(segs) > 1 and s.get("jumps"):
                for i in range(len(segs) - 1):
                    xa, ya = segs[i][-1]
                    xb, yb = segs[i + 1][0]
                    self.create_line(X(xa), Y(ya), X(xb), Y(yb), fill=color,
                                     dash=(3, 2), width=1)
                    for (cx, cy) in ((xa, ya), (xb, yb)):
                        r = 3
                        self.create_oval(X(cx) - r, Y(cy) - r, X(cx) + r, Y(cy) + r,
                                         outline=color, fill=PANEL, width=1)

        # legend of the overlay graph
        if self.legend:
            lx, ly = x0 + 8, y0 + 6
            self.create_rectangle(lx, ly, lx + 104, ly + 14 * len(self.series) + 6,
                                  fill="#fbfbf8", outline=BORDER)
            for i, s in enumerate(self.series):
                cy = ly + 12 + i * 14
                self.create_line(lx + 6, cy, lx + 24, cy,
                                 fill=s.get("color", self.accent), width=2,
                                 dash=(5, 3) if s.get("dashed") else None)
                self.create_text(lx + 30, cy, anchor="w",
                                 text=s.get("label", ""), font=FONT_SMALL, fill=INK)

        # current marker
        if self.marker:
            mx, my = self.marker["x"], self.marker["y"]
            if x_lo <= mx <= x_hi:
                px, py = X(mx), Y(my)
                self.create_line(px, py, px, y1, fill=FAINT, dash=(2, 2))
                if y_lo < 0.0 < y_hi:
                    self.create_line(px, py, x0, Y(0.0) if abs(my) > 1e-9 else py,
                                     fill=FAINT, dash=(2, 2))
                else:
                    self.create_line(px, py, x0, py, fill=FAINT, dash=(2, 2))
                r = 5 if self.marker.get("undefined") else 4
                if self.marker.get("undefined"):
                    self.create_oval(px - r, py - r, px + r, py + r,
                                     outline=self.accent, fill=PANEL, width=2)
                else:
                    self.create_oval(px - r, py - r, px + r, py + r,
                                     outline="#2b2b2b", fill=self.accent, width=1)
                label = self.marker.get("text", "")
                if label:
                    tw = 8 * len(label) + 12
                    bx = px - tw - 10 if px + tw + 10 > x1 else px + 10
                    # a box flipped to the left must not fall off the canvas
                    bx = max(x0, min(bx, w - tw - 2))
                    by = py - 22 if py - 22 > y0 else py + 8
                    self.create_rectangle(bx, by, bx + tw, by + 17,
                                          fill="#fdfcf8", outline=BORDER)
                    self.create_text(bx + 6, by + 8, anchor="w", text=label,
                                     font=MONO_SMALL, fill=INK)
            else:
                self.create_text(x1 - 6, y0 + 10, anchor="e",
                                 text=self.marker.get("off_text", ""),
                                 font=FONT_SMALL, fill="#b03040")
        if self.note:
            self.create_text(x0 + 8, y1 - 8, anchor="w", text=self.note,
                             font=FONT_SMALL, fill=FAINT)


# ---------------------------------------------------------------------------
# 9. Scrollable panel (used inside tabs only)
# ---------------------------------------------------------------------------


class ScrollFrame(tk.Frame):
    def __init__(self, master, bg=PANEL):
        tk.Frame.__init__(self, master, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._sync_region)
        self.canvas.bind("<Configure>", self._sync_width)
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        self.width = 200

    def _sync_region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_width(self, event):
        self.width = event.width
        self.canvas.itemconfigure(self._win, width=event.width)

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        for child in self.inner.winfo_children():
            child.destroy()


# ---------------------------------------------------------------------------
# 10. Application
# ---------------------------------------------------------------------------

DEFAULT_X = 1.0
DEFAULT_RANGE = 5.0


class App(object):
    def __init__(self, root, lang="ru"):
        self.root = root
        self.lang = lang
        self.fn_key = "sigmoid"
        self.x = DEFAULT_X
        self.params = {"alpha": PARAMS["alpha"]["default"],
                       "beta": PARAMS["beta"]["default"]}
        self.x_range = DEFAULT_RANGE
        self.cmp_mode = "f"
        self.help_key = HELP_IDS[0]
        self.status = tr(TEXTS[lang], "ui.status_ready")
        self._i18n = []
        self._suppress_slider = False
        self._suppress_param = False
        self._slider_echo = None
        self._param_echo = None
        self._entry_error = False

        root.title(tr(TEXTS[lang], "ui.title"))
        root.configure(bg=BG)
        root.geometry("1300x780")
        root.minsize(1024, 640)
        self._styles()

        self.body = tk.Frame(root, bg=BG)
        self.bar = tk.Frame(root, bg=BG)
        self.bar.pack(side="bottom", fill="x")
        self.body.pack(side="top", fill="both", expand=True)

        self.left = self._build_left(self.body)
        self.center = self._build_center(self.body)
        self.right = tk.Frame(self.body, bg=BG)

        self.body.columnconfigure(0, minsize=286, weight=0)
        self.body.columnconfigure(1, minsize=330, weight=1)
        self.body.columnconfigure(2, minsize=430, weight=0)
        self.body.rowconfigure(0, weight=1)
        self.left.grid(row=0, column=0, sticky="nswe", padx=(8, 4), pady=6)
        self.center.grid(row=0, column=1, sticky="nswe", padx=4, pady=6)
        self.right.grid(row=0, column=2, sticky="nswe", padx=(4, 8), pady=6)

        self._build_bar()
        self._build_tabs()
        self._apply_i18n()
        self.refresh()

    # -- localization -------------------------------------------------------
    def _apply_i18n(self):
        """Push the active language tree into every statically labeled widget."""
        T = TEXTS[self.lang]
        for (widget, path, _unused) in self._i18n:
            if widget.winfo_exists():
                widget.configure(text=tr(T, path))
        for (path, idx) in self._tab_paths:
            self.nb.tab(idx, text=tr(T, path))

    # -- styling ------------------------------------------------------------
    def _styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(2, 4, 2, 0))
        style.configure("TNotebook.Tab", font=FONT, padding=(9, 4),
                        background=BTN_BG, foreground=INK, bordercolor=BORDER)
        style.map("TNotebook.Tab",
                  background=[("selected", SEL_BG)],
                  foreground=[("selected", SEL_FG)])
        style.configure("TCombobox", font=FONT, arrowsize=14)
        style.configure("Vertical.TScrollbar", background=BTN_BG, troughcolor=BG,
                        arrowcolor=MUTED, bordercolor=BORDER)
        style.configure("Horizontal.TScrollbar", background=BTN_BG, troughcolor=BG,
                        arrowcolor=MUTED, bordercolor=BORDER)
        root_font = ("Segoe UI", 10)
        self.root.option_add("*TCombobox*Listbox.font", root_font)

    # -- small builders -----------------------------------------------------
    def _static(self, master, path, font=FONT_SECTION, fg=INK, anchor="w",
                justify="left", wrap=0, bg=BG):
        lbl = tk.Label(master, text=tr(TEXTS[self.lang], path), font=font, fg=fg,
                       bg=bg, anchor=anchor, justify=justify)
        if wrap:
            lbl.configure(wraplength=wrap)
        self._i18n.append((lbl, path, None))
        return lbl

    def _button(self, master, text, command, path=None, width=0, font=FONT,
                bg=BTN_BG, fg=INK):
        btn = tk.Button(master, text=text, command=command, font=font, bg=bg,
                        fg=fg, activebackground=BTN_ACTIVE, relief="flat",
                        bd=0, highlightthickness=1, highlightbackground=BORDER,
                        highlightcolor=SEL_BG, cursor="hand2", pady=3, padx=6)
        if width:
            btn.configure(width=width)
        if path:
            self._i18n.append((btn, path, None))
        return btn

    # -- left column --------------------------------------------------------
    def _build_left(self, parent):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        pad = dict(padx=10)
        self._static(frame, "ui.col_functions", font=FONT_TITLE,
                     fg=SEL_BG).grid(row=0, column=0, sticky="w", **pad, pady=(8, 2))
        self.fn_btns = {}
        self._left_wraps = []
        self._left_width = 0
        row = 1
        for key in ORDER:
            b = tk.Button(frame, text="", command=lambda k=key: self.set_function(k),
                          font=FONT, anchor="w", relief="flat", bd=0,
                          highlightthickness=1, highlightbackground=BORDER,
                          highlightcolor=SEL_BG, activebackground=BTN_ACTIVE,
                          cursor="hand2", padx=8, pady=3, justify="left")
            b.grid(row=row, column=0, sticky="we", padx=10, pady=1)
            self.fn_btns[key] = b
            row += 1

        sep = tk.Frame(frame, bg=BORDER, height=1)
        sep.grid(row=row, column=0, sticky="we", padx=10, pady=(6, 4))
        row += 1

        self._static(frame, "ui.col_input", fg=SEL_BG).grid(
            row=row, column=0, sticky="w", **pad); row += 1
        entry_row = tk.Frame(frame, bg=PANEL)
        entry_row.grid(row=row, column=0, sticky="we", **pad, pady=(2, 2)); row += 1
        self.x_entry = tk.Entry(entry_row, font=MONO_B, width=9, justify="right",
                                relief="flat", bd=1, highlightthickness=1,
                                highlightbackground=BORDER,
                                highlightcolor=SEL_BG, bg=PANEL, fg=INK)
        self.x_entry.grid(row=0, column=0, sticky="w")
        self.x_entry.bind("<Return>", lambda e: self.commit_x_entry())
        self.x_entry.bind("<FocusOut>", lambda e: self.commit_x_entry())
        self._button(entry_row, "-1", lambda: self.step_x(-1.0), width=4).grid(row=0, column=1, padx=(8, 2))
        self._button(entry_row, "+1", lambda: self.step_x(1.0), width=4).grid(row=0, column=2, padx=2)
        self._button(entry_row, "-0.1", lambda: self.step_x(-0.1), width=4).grid(row=0, column=3, padx=(8, 2))
        self._button(entry_row, "+0.1", lambda: self.step_x(0.1), width=4).grid(row=0, column=4, padx=2)
        self.x_slider = tk.Scale(frame, from_=-self.x_range, to=self.x_range,
                                 resolution=0.05, showvalue=0, orient="horizontal",
                                 bg=PANEL, troughcolor=BTN_ACTIVE, highlightthickness=1,
                                 highlightbackground=BORDER, sliderrelief="flat",
                                 sliderlength=26, length=250, font=FONT_SMALL,
                                 fg=INK, bd=0, command=self._slider_x)
        self.x_slider.grid(row=row, column=0, sticky="we", **pad, pady=(4, 0)); row += 1
        self._button(frame, "", self.reset_x, path="ui.btn_reset_x").grid(
            row=row, column=0, sticky="we", **pad, pady=(3, 4)); row += 1

        sep = tk.Frame(frame, bg=BORDER, height=1)
        sep.grid(row=row, column=0, sticky="we", padx=10, pady=(6, 4)); row += 1

        self._static(frame, "ui.col_params", fg=SEL_BG).grid(
            row=row, column=0, sticky="w", **pad); row += 1
        self.param_caption = tk.Label(frame, text="", font=MONO_B, fg=INK, bg=PANEL,
                                      anchor="w", justify="left")
        self.param_caption.grid(row=row, column=0, sticky="we", **pad, pady=(2, 1))
        row += 1
        self.param_row = tk.Frame(frame, bg=PANEL)
        self.param_row.grid(row=row, column=0, sticky="we", **pad); row += 1
        self.param_slider = tk.Scale(self.param_row, from_=0.0, to=1.0, resolution=0.01,
                                     showvalue=0, orient="horizontal", bg=PANEL,
                                     troughcolor=BTN_ACTIVE, highlightthickness=1,
                                     highlightbackground=BORDER, sliderrelief="flat",
                                     sliderlength=22, length=170, font=FONT_SMALL,
                                     command=self._slider_param)
        self.param_slider.pack(side="left", fill="x", expand=True)
        self.param_entry = tk.Entry(self.param_row, font=MONO, width=7, justify="right",
                                    relief="flat", bd=1, highlightthickness=1,
                                    highlightbackground=BORDER,
                                    highlightcolor=SEL_BG)
        self.param_entry.pack(side="left", padx=(6, 0))
        self.param_entry.bind("<Return>", lambda e: self.commit_param_entry())
        self.param_entry.bind("<FocusOut>", lambda e: self.commit_param_entry())
        self.param_hint = tk.Label(frame, text="", font=FONT_SMALL, fg=MUTED, bg=PANEL,
                                   anchor="w", justify="left", wraplength=262)
        self._left_wraps.append(self.param_hint)
        self.param_hint.grid(row=row, column=0, sticky="we", **pad, pady=(2, 2)); row += 1

        sep = tk.Frame(frame, bg=BORDER, height=1)
        sep.grid(row=row, column=0, sticky="we", padx=10, pady=(6, 4)); row += 1

        self._static(frame, "ui.col_view", fg=SEL_BG).grid(
            row=row, column=0, sticky="w", **pad); row += 1
        range_row = tk.Frame(frame, bg=PANEL)
        range_row.grid(row=row, column=0, sticky="we", **pad, pady=(2, 1)); row += 1
        self.range_btns = {}
        for i, span in enumerate(X_RANGES):
            b = tk.Button(range_row, text="+/- %g" % span,
                          command=lambda s=span: self.set_range(s), font=FONT,
                          relief="flat", bd=0, highlightthickness=1,
                          highlightbackground=BORDER, highlightcolor=SEL_BG,
                          activebackground=BTN_ACTIVE, cursor="hand2", pady=3)
            b.grid(row=0, column=i, sticky="we", padx=2)
            self.range_btns[span] = b
        range_row.columnconfigure((0, 1, 2, 3), weight=1, uniform="r")
        self.range_caption = tk.Label(frame, text="", font=FONT_SMALL, fg=MUTED,
                                      bg=PANEL, anchor="w")
        self.range_caption.grid(row=row, column=0, sticky="we", **pad); row += 1

        sep = tk.Frame(frame, bg=BORDER, height=1)
        sep.grid(row=row, column=0, sticky="we", padx=10, pady=(6, 4)); row += 1

        self._static(frame, "ui.col_presets", fg=SEL_BG).grid(
            row=row, column=0, sticky="w", **pad); row += 1
        self.exp_box = ttk.Combobox(frame, state="readonly", font=FONT)
        self.exp_box.grid(row=row, column=0, sticky="we", **pad, pady=(2, 2)); row += 1
        self.exp_box.bind("<<ComboboxSelected>>", self._exp_selected)
        self.exp_note = tk.Label(frame, text="", font=FONT_SMALL, fg=INK, bg=PANEL,
                                 anchor="w", justify="left", wraplength=262)
        self._left_wraps.append(self.exp_note)
        self.exp_note.grid(row=row, column=0, sticky="we", **pad, pady=(0, 4)); row += 1

        sep = tk.Frame(frame, bg=BORDER, height=1)
        sep.grid(row=row, column=0, sticky="we", padx=10, pady=(6, 4)); row += 1

        self._static(frame, "ui.col_language", fg=SEL_BG).grid(
            row=row, column=0, sticky="w", **pad); row += 1
        ctl = tk.Frame(frame, bg=PANEL)
        ctl.grid(row=row, column=0, sticky="we", **pad, pady=(2, 1)); row += 1
        self.lang_btn = tk.Button(ctl, text="", command=self.toggle_language,
                                  font=FONT_BOLD,
                                  relief="flat", bd=0, highlightthickness=1,
                                  highlightbackground=BORDER, highlightcolor=SEL_BG,
                                  activebackground=BTN_ACTIVE, cursor="hand2",
                                  width=5, pady=3, bg=BTN_BG)
        self.lang_btn.pack(side="left")
        self._button(ctl, "", self.reset_all, path="ui.btn_reset_all").pack(
            side="left", padx=(6, 0))
        lang_hint = self._static(frame, "ui.lang_hint", font=FONT_SMALL,
                                 fg=MUTED, wrap=262)
        self._left_wraps.append(lang_hint)
        lang_hint.grid(row=row, column=0, sticky="we", **pad, pady=(0, 6))
        row += 1
        frame.columnconfigure(0, weight=1)
        frame.bind("<Configure>", self._left_resized)
        return frame

    # -- center column ------------------------------------------------------
    def _build_center(self, parent):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground=BORDER)
        head = tk.Frame(frame, bg=PANEL)
        head.pack(fill="x", padx=12, pady=(8, 2))
        self.fn_name = tk.Label(head, text="", font=FONT_TITLE, fg=INK, bg=PANEL,
                                anchor="w")
        self.fn_name.pack(side="left")
        self.fn_tag = tk.Label(head, text="", font=FONT_TAG, fg=MUTED, bg=PANEL,
                               anchor="e", justify="right")
        self.fn_tag.pack(side="right")
        self.fn_formula = tk.Label(frame, text="", font=MONO, fg=SEL_BG, bg=PANEL,
                                   anchor="w", justify="left")
        self.fn_formula.pack(fill="x", padx=12, pady=(0, 4))

        self.f_canvas = PlotCanvas(frame, F_COLOR, "f")
        self.f_canvas.pack(fill="both", expand=True, padx=12, pady=(2, 3))
        self.d_canvas = PlotCanvas(frame, D_COLOR, "df")
        self.d_canvas.pack(fill="both", expand=True, padx=12, pady=(3, 4))

        readout = tk.Frame(frame, bg=BTN_BG, highlightthickness=1,
                           highlightbackground=BORDER)
        readout.pack(fill="x", padx=12, pady=(0, 10))
        self.ro_x = tk.Label(readout, text="", font=MONO_B, bg=BTN_BG, fg=INK)
        self.ro_f = tk.Label(readout, text="", font=MONO_B, bg=BTN_BG, fg=F_COLOR)
        self.ro_d = tk.Label(readout, text="", font=MONO_B, bg=BTN_BG, fg=D_COLOR)
        for i, w in enumerate((self.ro_x, self.ro_f, self.ro_d)):
            w.grid(row=0, column=i, sticky="w", padx=14, pady=4)
        readout.columnconfigure(3, weight=1)
        return frame

    def _build_bar(self):
        self.status_lbl = tk.Label(self.bar, text=self.status, font=FONT_SMALL,
                                   fg=MUTED, bg=BG, anchor="w")
        self.status_lbl.pack(side="left", padx=(12, 6), pady=(2, 6))
        # Packed first with side=right, so the signature ends up in the very
        # bottom-right corner and the hint sits to its left.
        tk.Label(self.bar, text=AUTHOR, font=FONT_BOLD, fg=MUTED, bg=BG,
                 anchor="e").pack(side="right", padx=(6, 12), pady=(2, 6))
        self._static(self.bar, "ui.sizes_hint", font=FONT_SMALL, fg=FAINT,
                     bg=BG).pack(side="right", padx=(6, 8), pady=(2, 6))

    # -- tabs ---------------------------------------------------------------
    def _build_tabs(self):
        self.nb = ttk.Notebook(self.right)
        self.nb.pack(fill="both", expand=True)
        T = TEXTS[self.lang]
        self.calc_sf = ScrollFrame(self.nb)
        self.cmp_sf = ScrollFrame(self.nb)
        self.phen_sf = ScrollFrame(self.nb)
        self.help_frame = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(self.calc_sf, text=tr(T, "ui.tab_calc"))
        self.nb.add(self.cmp_sf, text=tr(T, "ui.tab_compare"))
        self.nb.add(self.phen_sf, text=tr(T, "ui.tab_phen"))
        self.nb.add(self.help_frame, text=tr(T, "ui.tab_help"))
        self._tab_paths = [("ui.tab_calc", 0), ("ui.tab_compare", 1),
                           ("ui.tab_phen", 2), ("ui.tab_help", 3)]
        self._tab_width = 420
        for sf in (self.calc_sf, self.cmp_sf, self.phen_sf):
            sf.canvas.bind("<Configure>", self._note_tab_width)

        # help tab: chooser on top, text below
        head = tk.Frame(self.help_frame, bg=PANEL)
        head.grid(row=0, column=0, sticky="we", padx=8, pady=6)
        # the width has to hold the longest section title: the Russian one needs
        # 277 px, and the row itself is 507 px in every supported window size
        self.help_pick = ttk.Combobox(head, state="readonly", font=FONT, width=48)
        self.help_pick.pack(side="left")
        self.help_pick.bind("<<ComboboxSelected>>", self._help_selected)
        self._button(head, "", self._help_prev, path="help.prev").pack(
            side="right", padx=(4, 0))
        self._button(head, "", self._help_next, path="help.next").pack(
            side="right", padx=(4, 0))
        body = tk.Frame(self.help_frame, bg=PANEL)
        body.grid(row=1, column=0, sticky="nswe", padx=(8, 2), pady=(0, 8))
        self.help_index = tk.Label(self.help_frame, text="", font=FONT_SMALL,
                                   fg=MUTED, bg=PANEL, anchor="w")
        self.help_index.grid(row=2, column=0, sticky="we", padx=10, pady=(0, 6))
        self.help_text_w = tk.Text(body, font=FONT, wrap="word", relief="flat", bd=0,
                                   highlightthickness=1, highlightbackground=BORDER,
                                   bg=PANEL, fg=INK, padx=8, pady=6, spacing1=4,
                                   spacing3=4)
        self.help_scroll = ttk.Scrollbar(body, orient="vertical",
                                         command=self.help_text_w.yview)
        self.help_text_w.configure(yscrollcommand=self.help_scroll.set)
        self.help_text_w.pack(side="left", fill="both", expand=True)
        self.help_scroll.pack(side="right", fill="y")
        self.help_text_w.configure(state="disabled")
        self.help_frame.columnconfigure(0, weight=1)
        self.help_frame.rowconfigure(1, weight=1)

    def _note_tab_width(self, event):
        if abs(event.width - self._tab_width) > 10:
            self._tab_width = event.width
            self.root.after(120, self.refresh)

    def _wrap(self):
        return max(150, self._tab_width - 34)

    def _left_wrap(self):
        return max(180, self._left_width - 24)

    def _left_resized(self, event):
        """The wrapped hints of the left column follow the real column width."""
        if abs(event.width - self._left_width) <= 10:
            return
        self._left_width = event.width
        wrap = max(180, event.width - 24)
        for lbl in self._left_wraps:
            lbl.configure(wraplength=wrap)

    # -- calculation tab ----------------------------------------------------
    def _refresh_calc(self):
        act = BY_KEY[self.fn_key]
        T = TEXTS[self.lang]
        inner = self.calc_sf.inner
        for c in inner.winfo_children():
            c.destroy()
        lines = calc_block(act, self.x, self.params, self.lang)

        def grouped(role):
            return [text for (r, text) in lines if r == role]

        blocks = [
            ("calc.formula", [act.formula], True, False),
            ("calc.substitution", grouped("substitution"), True, False),
            ("calc.result", grouped("result"), True, False),
            ("calc.dformula", [act.dformula], True, False),
            ("calc.dsubstitution", grouped("dsubstitution"), True, False),
            ("calc.dresult", grouped("dresult"), True, False),
            ("calc.note", grouped("note"), False, True),
            ("calc.kink", grouped("kink"), False, True),
            ("calc.meaning", meaning_lines(act, self.x, self.params, self.lang),
             False, True),
        ]
        row = 0
        for (caption, texts, mono, prose) in blocks:
            texts = [t for t in texts if t]
            if not texts:
                continue
            tk.Label(inner, text=tr(T, caption), font=FONT_SECTION, fg=SEL_BG,
                     bg=PANEL, anchor="w").grid(row=row, column=0, sticky="w",
                                               pady=(8, 1))
            row += 1
            for t in texts:
                if mono:
                    w = tk.Label(inner, text=t, font=MONO, fg=INK, bg=BTN_BG,
                                 anchor="w", justify="left", padx=6, pady=2)
                    w.grid(row=row, column=0, sticky="we", pady=1)
                else:
                    w = tk.Label(inner, text=t, font=FONT, fg=INK, bg=PANEL,
                                 anchor="w", justify="left",
                                 wraplength=self._wrap())
                    w.grid(row=row, column=0, sticky="we", pady=1)
                row += 1
        inner.columnconfigure(0, weight=1)

    # -- comparison tab -----------------------------------------------------
    def _refresh_compare(self):
        act = BY_KEY[self.fn_key]
        T = TEXTS[self.lang]
        inner = self.cmp_sf.inner
        for c in inner.winfo_children():
            c.destroy()
        wrap = self._wrap()
        tk.Label(inner, text=tr(T, "cmp.title") % fmt(self.x, 2), font=FONT_TITLE,
                 fg=SEL_BG, bg=PANEL, anchor="w").grid(row=0, column=0, sticky="w",
                                                       pady=(6, 0))
        tk.Label(inner, text=tr(T, "cmp.hint"), font=FONT_SMALL, fg=MUTED, bg=PANEL,
                 anchor="w", justify="left", wraplength=wrap).grid(
            row=1, column=0, sticky="w", pady=(0, 6))

        head = tk.Frame(inner, bg=BTN_BG)
        head.grid(row=2, column=0, sticky="we", pady=(0, 2))
        for col, path in enumerate(("cmp.h_function", "cmp.h_fx", "cmp.h_dfx",
                                    "cmp.h_range")):
            tk.Label(head, text=tr(T, path), font=FONT_BOLD, fg=INK, bg=BTN_BG,
                     anchor="w" if col == 0 else "center").grid(
                row=0, column=col, sticky="we", padx=6, pady=3)
        head.columnconfigure(0, minsize=112, weight=3)
        head.columnconfigure(1, minsize=74, weight=2)
        head.columnconfigure(2, minsize=74, weight=2)
        head.columnconfigure(3, minsize=84, weight=2)
        row = 3
        for i, key in enumerate(ORDER):
            item = BY_KEY[key]
            color = LEGEND_COLORS[i]
            v = item.f(self.x, self.params)
            d = item.df(self.x, self.params)
            block = tk.Frame(inner, bg=PANEL if key != self.fn_key else "#eef4f9")
            block.grid(row=row, column=0, sticky="we", pady=(0, 1))
            row += 1
            line = tk.Frame(block, bg=block["bg"])
            line.grid(row=0, column=0, columnspan=4, sticky="we")
            tk.Label(line, text=tr(T, "fn.%s.name" % key), font=FONT_BOLD, fg=color,
                     bg=block["bg"], anchor="w").grid(row=0, column=0, sticky="we",
                                                      padx=6, pady=3)
            tk.Label(line, text=fmt(v, 4), font=MONO, fg=INK, bg=block["bg"],
                     anchor="center").grid(row=0, column=1, padx=6)
            dtext = "-" if item.is_kink(self.x) else fmt(d, 4)
            tk.Label(line, text=dtext, font=MONO, fg=INK, bg=block["bg"],
                     anchor="center").grid(row=0, column=2, padx=6)
            tk.Label(line, text=item.out_range, font=MONO_SMALL, fg=MUTED,
                     bg=block["bg"], anchor="center").grid(row=0, column=3, padx=6)
            line.columnconfigure(0, minsize=112, weight=3)
            line.columnconfigure(1, minsize=74, weight=2)
            line.columnconfigure(2, minsize=74, weight=2)
            line.columnconfigure(3, minsize=84, weight=2)
            for path, field in (("cmp.h_deriv", "deriv_behavior"),
                                ("cmp.h_char", "characteristic"),
                                ("cmp.h_limit", "limitation")):
                cap = tr(T, path)
                txt = "%s: %s" % (cap, tr(T, "fn.%s.%s" % (key, field)))
                tk.Label(block, text=txt, font=FONT_SMALL, fg=INK, bg=block["bg"],
                         anchor="w", justify="left",
                         wraplength=max(120, wrap - len(cap) - 6)).grid(
                    row=1 if path == "cmp.h_deriv" else (2 if path == "cmp.h_char" else 3),
                    column=0, columnspan=4, sticky="we", padx=6, pady=1)
            tk.Frame(inner, bg=BORDER, height=1).grid(row=row, column=0, sticky="we")
            row += 1

        # visual comparison
        ctl = tk.Frame(inner, bg=PANEL)
        ctl.grid(row=row, column=0, sticky="we", pady=(10, 4))
        row += 1
        tk.Label(ctl, text=tr(T, "cmp.visual_title"), font=FONT_SECTION, fg=SEL_BG,
                 bg=PANEL, anchor="w").pack(side="left")
        self.cmp_btns = {}
        for mode, path in (("f", "cmp.show_f"), ("df", "cmp.show_df")):
            b = tk.Button(ctl, text=tr(T, path), command=lambda m=mode: self.set_cmp_mode(m),
                          font=FONT, relief="flat", bd=0, highlightthickness=1,
                          highlightbackground=BORDER, highlightcolor=SEL_BG,
                          activebackground=BTN_ACTIVE, cursor="hand2", padx=8, pady=2)
            b.pack(side="right", padx=(4, 0))
            self.cmp_btns[mode] = b
        self.cmp_canvas = PlotCanvas(inner, F_COLOR, "cmp", height=200)
        self.cmp_canvas.grid(row=row, column=0, sticky="we", pady=(0, 4))
        row += 1
        tk.Label(inner, text=tr(T, "cmp.visual_hint"), font=FONT_SMALL, fg=MUTED,
                 bg=PANEL, anchor="w", justify="left",
                 wraplength=wrap).grid(row=row, column=0, sticky="w", pady=(0, 10))
        self._draw_compare()
        inner.columnconfigure(0, weight=1)

    def _draw_compare(self):
        if not getattr(self, "cmp_canvas", None):
            return
        T = TEXTS[self.lang]
        lo, hi = -self.x_range, self.x_range
        series = []
        for i, key in enumerate(ORDER):
            item = BY_KEY[key]
            pts, disc = build_curve(item, self.params, lo, hi,
                                    use_deriv=(self.cmp_mode == "df"), samples=140)
            series.append({"pts": pts, "color": LEGEND_COLORS[i],
                           "label": tr(T, "legend.%s" % key), "jumps": disc})
        use_deriv = self.cmp_mode == "df"
        title = tr(T, "graph.df_title" if use_deriv else "graph.f_title")
        self.cmp_canvas.set_series(series, (lo, hi), marker=None, legend=True,
                                   title=title,
                                   xlabel=tr(T, "graph.axis_x"),
                                   ylabel=tr(T, "graph.axis_y_d" if use_deriv
                                            else "graph.axis_y"))
        self._mark_compare_x(lo, hi)

    def _mark_compare_x(self, lo, hi):
        """Vertical line at the current x on the overlay graph."""
        c = self.cmp_canvas
        if not (lo <= self.x <= hi):
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 80:
            return
        x0, x1 = 54, w - 16
        px = x0 + (self.x - lo) * (x1 - x0) / (hi - lo)
        c.create_line(px, 26, px, h - 24, fill="#b03040", dash=(3, 2), width=1)

    # -- phenomena tab ------------------------------------------------------
    def _refresh_phen(self):
        T = TEXTS[self.lang]
        inner = self.phen_sf.inner
        for c in inner.winfo_children():
            c.destroy()
        tk.Label(inner, text=tr(T, "phen.title"), font=FONT_TITLE, fg=SEL_BG,
                 bg=PANEL, anchor="w").grid(row=0, column=0, sticky="w", pady=(6, 0))
        tk.Label(inner, text=tr(T, "phen.hint"), font=FONT_SMALL, fg=MUTED, bg=PANEL,
                 anchor="w", justify="left",
                 wraplength=self._wrap()).grid(row=1, column=0, sticky="w", pady=(0, 8))
        row = 2
        for i, key in enumerate(ORDER):
            color = LEGEND_COLORS[i]
            block = tk.Frame(inner, bg=PANEL)
            block.grid(row=row, column=0, sticky="we", pady=(2, 6))
            row += 1
            head = tk.Frame(block, bg=PANEL)
            head.pack(fill="x")
            tk.Label(head, text=tr(T, "fn.%s.name" % key), font=FONT_BOLD, fg=color,
                     bg=PANEL, anchor="w").pack(side="left")
            self._button(head, tr(T, "phen.action"),
                         lambda k=key: self.show_phenomenon(k)).pack(side="right")
            tk.Label(block, text=tr(T, "fn.%s.explanation" % key), font=FONT,
                     fg=INK, bg=PANEL, anchor="w", justify="left",
                     wraplength=self._wrap()).pack(fill="x", pady=(2, 0))
            tk.Frame(inner, bg=BORDER, height=1).grid(row=row, column=0, sticky="we")
            row += 1
        inner.columnconfigure(0, weight=1)

    # -- help tab -----------------------------------------------------------
    def _refresh_help(self):
        T = TEXTS[self.lang]
        titles = [tr(T, "help_text.%s" % k)[0] for k in HELP_IDS]
        self.help_pick["values"] = titles
        idx = HELP_IDS.index(self.help_key)
        self.help_pick.current(idx)
        title, paras = help_text(self.help_key, self.lang)
        widget = self.help_text_w
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", title + "\n\n")
        for p in paras:
            widget.insert("end", p + "\n\n")
        widget.tag_configure("h", font=FONT_TITLE, foreground=SEL_BG)
        widget.tag_add("h", "1.0", "1.end")
        widget.configure(state="disabled")
        widget.yview_moveto(0)
        self.help_index.configure(text=tr(T, "help.index") % len(HELP_IDS))

    def _help_selected(self, event=None):
        self.help_key = HELP_IDS[self.help_pick.current()]
        self._refresh_help()

    def _help_step(self, delta):
        i = HELP_IDS.index(self.help_key) + delta
        self.help_key = HELP_IDS[int(i) % len(HELP_IDS)]
        self._refresh_help()

    def _help_next(self):
        self._help_step(1)

    def _help_prev(self):
        self._help_step(-1)

    # -- interactions -------------------------------------------------------
    def set_function(self, key):
        self.fn_key = key
        self.status = tr(TEXTS[self.lang], "status.func_set") % \
            tr(TEXTS[self.lang], "fn.%s.name" % key)
        self.refresh()

    def set_cmp_mode(self, mode):
        self.cmp_mode = mode
        self._refresh_compare()

    def set_x(self, value):
        self.x = float(value)
        self.refresh()

    def step_x(self, delta):
        value = round(self.x + delta, 4)
        if abs(value) > MAX_ABS_X:
            self.status = tr(TEXTS[self.lang], "status.invalid") % \
                tr(TEXTS[self.lang], "val.too_big")
            return self.refresh()
        self.x = value
        self.refresh()

    def commit_x_entry(self):
        raw = self.x_entry.get()
        value, error = parse_x(raw)
        T = TEXTS[self.lang]
        if error:
            self._entry_error = True
            self.x_entry.configure(bg=ERR_BG)
            self.status = tr(T, "status.invalid") % tr(T, error)
            return self.refresh()
        self._entry_error = False
        self.x_entry.configure(bg=OK_BG)
        self.x = value
        self.refresh()

    def _slider_x(self, raw):
        if self._suppress_slider:
            return
        try:
            value = float(raw)
        except ValueError:
            return
        # tk.Scale echoes a command for our own set()/range change and rounds it
        # to its resolution; only a value off that grid is a real user drag.
        tol = 0.5 * float(self.x_slider.cget("resolution")) + 1e-9
        if abs(value - self.x) <= tol:
            return
        if self._slider_echo is not None and abs(value - self._slider_echo) <= tol:
            return
        self.x = value
        self._entry_error = False
        self.refresh()

    def set_range(self, span):
        if span not in X_RANGES:
            return
        self.x_range = float(span)
        self.status = tr(TEXTS[self.lang], "status.range_set") % (
            fmt(-span, 0), fmt(span, 0))
        self.refresh()

    def _slider_param(self, raw):
        if self._suppress_param:
            return
        act = BY_KEY[self.fn_key]
        if not act.has_param():
            return
        try:
            value = float(raw)
        except ValueError:
            return
        tol = 0.5 * float(self.param_slider.cget("resolution")) + 1e-9
        if abs(value - self.params[act.param]) <= tol:
            return
        if self._param_echo is not None and abs(value - self._param_echo) <= tol:
            return
        self.params[act.param] = value
        self.status = tr(TEXTS[self.lang], "status.param_set") % (act.param, fmt(value, 3))
        self.refresh()

    def commit_param_entry(self):
        act = BY_KEY[self.fn_key]
        if not act.has_param():
            return
        spec = PARAMS[act.param]
        try:
            value = float(self.param_entry.get().strip().replace(",", "."))
        except ValueError:
            value = None
        reason = None
        if value is None:
            reason = tr(TEXTS[self.lang], "val.number")
        elif value < spec["min"] - 1e-9 or value > spec["max"] + 1e-9:
            reason = tr(TEXTS[self.lang], "val.param")
        if reason is not None:
            self.param_entry.configure(bg=ERR_BG)
            self.status = tr(TEXTS[self.lang], "status.param_bad") % (
                act.param, reason)
            return self.refresh()
        self.param_entry.configure(bg=OK_BG)
        self.params[act.param] = value
        self.status = tr(TEXTS[self.lang], "status.param_set") % (
            act.param, fmt(value, 3))
        self.refresh()

    def reset_x(self):
        self.x = DEFAULT_X
        self._entry_error = False
        self.status = tr(TEXTS[self.lang], "status.reset_x")
        self.refresh()

    def reset_all(self):
        self.fn_key = "sigmoid"
        self.x = DEFAULT_X
        self.x_range = DEFAULT_RANGE
        self.params = {"alpha": PARAMS["alpha"]["default"],
                       "beta": PARAMS["beta"]["default"]}
        self.cmp_mode = "f"
        self.help_key = HELP_IDS[0]
        self._entry_error = False
        self.status = tr(TEXTS[self.lang], "status.reset_all")
        self.refresh()

    def apply_state(self, st):
        act = BY_KEY[st["fn"]]
        self.fn_key = st["fn"]
        self.x = float(st["x"])
        self.x_range = float(st["rng"])
        if "alpha" in st:
            self.params["alpha"] = float(st["alpha"])
        if "beta" in st:
            self.params["beta"] = float(st["beta"])
        if "tab" in st:
            for i, (path, _idx) in enumerate(self._tab_paths):
                if path == "ui.tab_" + st["tab"]:
                    self.nb.select(self.nb.tabs()[i])
        self._entry_error = False
        self.refresh()

    def apply_experiment(self, exp_id):
        st = dict(EXPERIMENTS_BY_KEY[exp_id])
        # The status is written before the repaint, otherwise the status bar
        # would keep showing the previous action.
        name = exp_text(exp_id, self.lang)[0]
        self.status = tr(TEXTS[self.lang], "status.exp_applied") % name
        self.apply_state(st)

    def _exp_selected(self, event=None):
        idx = self.exp_box.current()
        if 0 <= idx < len(EXPERIMENTS):
            self.apply_experiment(EXPERIMENTS[idx][0])

    def show_phenomenon(self, key):
        name = tr(TEXTS[self.lang], "fn.%s.name" % key)
        self.status = tr(TEXTS[self.lang], "status.phen_applied") % name
        self.apply_state(dict(PHENOMENON_BY_KEY[key]))

    def toggle_language(self):
        self.set_language("en" if self.lang == "ru" else "ru")

    def set_language(self, lang):
        if lang not in TEXTS or lang == self.lang:
            return
        selected = self.nb.index(self.nb.select()) if self.nb.winfo_children() else 0
        self.lang = lang
        self.root.title(tr(TEXTS[lang], "ui.title"))
        self._apply_i18n()
        self.status = tr(TEXTS[lang], "status.lang_set")
        self.refresh()
        self.nb.select(selected)

    # -- refresh ------------------------------------------------------------
    def refresh(self):
        T = TEXTS[self.lang]
        act = BY_KEY[self.fn_key]
        v = act.f(self.x, self.params)
        d = act.df(self.x, self.params)

        self.fn_name.configure(text=tr(T, "fn.%s.name" % self.fn_key))
        self.fn_tag.configure(text=tr(T, "fn.%s.tagline" % self.fn_key))
        self.fn_formula.configure(text=act.formula)

        for key, btn in self.fn_btns.items():
            sel = key == self.fn_key
            btn.configure(text=tr(T, "fn.%s.name" % key),
                          bg=SEL_BG if sel else BTN_BG,
                          fg=SEL_FG if sel else INK,
                          highlightbackground=BORDER if sel else BORDER)
        self.lang_btn.configure(text=tr(T, "ui.lang_switch"))
        for span, btn in self.range_btns.items():
            sel = abs(span - self.x_range) < 1e-9
            btn.configure(bg=SEL_BG if sel else BTN_BG, fg=SEL_FG if sel else INK)
        self.range_caption.configure(text=tr(T, "ui.range_hint") %
                                     (fmt(-self.x_range, 0), fmt(self.x_range, 0)))

        # x slider + entry
        self._suppress_slider = True
        self._slider_echo = clamp(self.x, -self.x_range, self.x_range)
        self.x_slider.configure(from_=-self.x_range, to=self.x_range)
        self.x_slider.set(self._slider_echo)
        self._suppress_slider = False
        # only commit_x_entry lifts the error tint, so every other path that
        # accepts a value has to lift it here
        if not self._entry_error:
            self.x_entry.configure(bg=OK_BG)
        if not self._entry_error and self.x_entry.focus_get() is not self.x_entry:
            self.x_entry.delete(0, "end")
            self.x_entry.insert(0, fmt(self.x, 2))
        self.ro_x.configure(text="x = %s" % fmt(self.x, 2))
        self.ro_f.configure(text="f(x) = %s" % fmt(v, 6))
        self.ro_d.configure(text="f'(x) = %s" % ("-" if act.is_kink(self.x) else fmt(d, 6)))

        # parameter widgets
        if act.has_param():
            spec = PARAMS[act.param]
            self._suppress_param = True
            self._param_echo = clamp(self.params[act.param], spec["min"], spec["max"])
            self.param_slider.configure(from_=spec["min"], to=spec["max"],
                                        resolution=spec["step"])
            self.param_slider.set(self._param_echo)
            self._suppress_param = False
            self.param_caption.configure(text="%s = %s" % (act.param,
                                     fmt(self.params[act.param], 3)))
            self.param_hint.configure(text=tr(T, "ui.param_%s" % act.param))
            if self.param_entry.focus_get() is not self.param_entry:
                self.param_entry.delete(0, "end")
                self.param_entry.insert(0, fmt(self.params[act.param], 3))
                self.param_entry.configure(bg=OK_BG)
            self.param_row.grid()
        else:
            self.param_caption.configure(text=tr(T, "ui.no_params"))
            self.param_hint.configure(text="")
            self.param_row.grid_remove()

        self._refresh_graphs()

        # experiments
        self.exp_box["values"] = [exp_text(k, self.lang)[0] for (k, _s) in EXPERIMENTS]
        if self.exp_box.current() < 0:
            self.exp_box.current(0)
        note = exp_text(EXPERIMENTS[self.exp_box.current()][0], self.lang)[1]
        self.exp_note.configure(text=note, wraplength=self._left_wrap())

        self._refresh_calc()
        self._refresh_compare()
        self._refresh_phen()
        self._refresh_help()
        self.status_lbl.configure(text=self.status, fg="#b03040"
                                  if self._entry_error else MUTED)

    def _refresh_graphs(self):
        if not (self.f_canvas.winfo_exists() and self.d_canvas.winfo_exists()):
            return
        T = TEXTS[self.lang]
        act = BY_KEY[self.fn_key]
        lo, hi = -self.x_range, self.x_range
        v = act.f(self.x, self.params)
        d = act.df(self.x, self.params)
        pts, disc = build_curve(act, self.params, lo, hi)
        dpts, ddisc = build_curve(act, self.params, lo, hi, use_deriv=True)
        off = tr(T, "graph.out_of_view")
        m_f = {"x": self.x, "y": v, "off_text": off,
               "text": tr(T, "graph.point") % (fmt(self.x, 2), fmt(v, 4))}
        undef = act.is_kink(self.x)
        if undef:
            label = tr(T, "graph.point_d_undef") % fmt(self.x, 2)
        else:
            label = tr(T, "graph.point_d") % (fmt(self.x, 2), fmt(d, 4))
        m_d = {"x": self.x, "y": d, "off_text": off, "text": label,
               "undefined": undef}
        wide = self.center.winfo_width() > 430
        xname = tr(T, "graph.axis_x")
        self.f_canvas.set_series([{"pts": pts, "color": F_COLOR, "jumps": disc}],
                                 (lo, hi), marker=m_f,
                                 title=tr(T, "graph.f_title"),
                                 subtitle=act.formula if wide else "",
                                 xlabel=xname, ylabel=tr(T, "graph.axis_y"))
        flat = (not dpts) or max(abs(p[1]) for p in dpts) < 1e-12
        self.d_canvas.set_series([{"pts": dpts, "color": D_COLOR, "jumps": ddisc}],
                                 (lo, hi), marker=m_d,
                                 title=tr(T, "graph.df_title"),
                                 subtitle=act.dformula if wide else "",
                                 xlabel=xname, ylabel=tr(T, "graph.axis_y_d"),
                                 note=tr(T, "graph.flat_note") if flat else "")
        if abs(self.x) > 1e-12 and (self.x < lo or self.x > hi):
            self.status = tr(T, "status.out_of_range") % (
                fmt(self.x, 2), fmt(lo, 0), fmt(hi, 0))

    # -- main loop ----------------------------------------------------------


def parse_x(raw):
    """Return (value, error_key). Accepts comma and dot separators."""
    s = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not s:
        return None, "val.empty"
    try:
        value = float(s)
    except ValueError:
        return None, "val.number"
    if value != value or value in (float("inf"), float("-inf")):
        return None, "val.number"
    if abs(value) > MAX_ABS_X:
        return None, "val.too_big"
    return value, None


EXPERIMENTS = (
    ("normal", {"fn": "sigmoid", "x": 1.0, "rng": 5.0}),
    ("large_neg", {"fn": "sigmoid", "x": -8.0, "rng": 10.0}),
    ("large_pos", {"fn": "sigmoid", "x": 8.0, "rng": 10.0}),
    ("near_zero", {"fn": "tanh", "x": 0.1, "rng": 2.0}),
    ("relu_neg", {"fn": "relu", "x": -3.0, "rng": 5.0}),
    ("relu_pos", {"fn": "relu", "x": 3.0, "rng": 5.0}),
    ("relu_vs_leaky", {"fn": "leaky_relu", "x": -2.0, "rng": 5.0, "alpha": 0.10,
                       "tab": "compare"}),
    ("sigmoid_sat", {"fn": "sigmoid", "x": 6.0, "rng": 10.0}),
    ("tanh_sat", {"fn": "tanh", "x": -5.0, "rng": 10.0}),
)
EXPERIMENTS_BY_KEY = dict(EXPERIMENTS)

PHENOMENA = (
    ("sigmoid", {"fn": "sigmoid", "x": 6.0, "rng": 10.0}),
    ("tanh", {"fn": "tanh", "x": -5.0, "rng": 10.0}),
    ("relu", {"fn": "relu", "x": -3.0, "rng": 5.0}),
    ("leaky_relu", {"fn": "leaky_relu", "x": -2.0, "rng": 5.0, "alpha": 0.10}),
    ("binary_step", {"fn": "binary_step", "x": 0.0, "rng": 2.0}),
    ("softplus", {"fn": "softplus", "x": 0.0, "rng": 5.0, "beta": 1.0}),
)
PHENOMENON_BY_KEY = dict(PHENOMENA)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()


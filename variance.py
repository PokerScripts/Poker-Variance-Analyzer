#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
variance.py — анализ дисперсии результатов в покере (cash / MTT).
- Загрузка CSV/JSON или ввод параметров напрямую.
- Базовые метрики (mean/median/std, CI).
- Оценка вероятности даунстрика и риска разорения (RoR).
- Монте-Карло симуляции с нормальным или "толстохвостым" (t) шумом.
- Сохранение артефактов (CSV/JSON) и опциональный график.

Зависимости: Python 3.9+, numpy, pandas, matplotlib (опционально для графиков)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

# pandas и matplotlib опциональны: код аккуратно предупредит, если их нет
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # тип: ignore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # тип: ignore


# ============================
# Вспомогательные типы/структуры
# ============================

@dataclass
class ModeContext:
    mode: str  # "cash" | "mtt"
    units: str  # "bb" | "bi" | "$"
    # Для cash:
    wr_bb100: Optional[float] = None
    sd_bb100: Optional[float] = None
    # Для mtt:
    roi: Optional[float] = None          # в бай-инах, т.е. 0.1 = 10% ROI
    sd_bi: Optional[float] = None        # σ результата в би за турнир
    # Конвертация:
    bb_value: Optional[float] = None     # размер BB в валюте, если нужно
    avg_buyin: Optional[float] = None    # средний buy-in в валюте (для отчётов)
    # Горизонт (шаги моделирования):
    horizon_steps: int = 0               # cash: количество "100 рук" или сессий; mtt: турниры
    step_kind: str = ""                  # "per100" | "sessions" | "tournaments"


@dataclass
class Basics:
    n_obs: int
    mean: float
    median: float
    std: float
    cv: float
    ci_low: float
    ci_high: float
    # cash:
    wr_bb100: Optional[float] = None
    sd_bb100: Optional[float] = None
    # mtt:
    roi: Optional[float] = None
    sd_bi: Optional[float] = None


@dataclass
class RiskMetrics:
    p_down_ge_threshold: Optional[float]
    risk_of_ruin_approx: Optional[float]
    p_target: Optional[float]
    p_loss: Optional[float]


@dataclass
class SimResult:
    final_results: np.ndarray
    drawdowns: Optional[np.ndarray]
    bankruptcies: Optional[np.ndarray]
    p10: float
    p50: float
    p90: float
    risk_of_ruin_mc: float
    p_hit_target: Optional[float]
    p_hit_drawdown: Optional[float]
    sample_trajectories: Optional[np.ndarray]  # shape: (k, steps) в единицах модели


# ============================
# Парсинг аргументов
# ============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Анализ дисперсии результатов в покере (cash/MTT) с Монте-Карло симуляциями.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Ввод данных/режим
    p.add_argument("--input", type=str, help="Путь к CSV-файлу с данными.")
    p.add_argument("--json-in", type=str, help="Путь к JSON-файлу с данными (массив объектов).")
    p.add_argument("--mode", type=str, choices=["cash", "mtt"], help="Принудительный режим анализа.")

    # Конвертация/единицы
    p.add_argument("--currency", type=str, help="Код валюты (для подписи).", default="")
    p.add_argument("--bb", type=str, help="Размер большого блайнда в валюте, например '1$' или просто '1'.")
    p.add_argument("--avg_buyin", type=str, help="Средний buy-in в валюте, например '55$' или '55'.")

    # Горизонт
    p.add_argument("--horizon", type=str, help="Горизонт: '5000h'| '50s' | '100t'.", default="")
    # Параметры порогов/целей
    p.add_argument("--drawdown", type=str, help="Порог просадки: например '20bi', '1000bb', '500$'.")
    p.add_argument("--target", type=float, help="Целевая прибыль в текущих единицах (bb/bi/$).")

    # Монте-Карло
    p.add_argument("--nsims", type=int, default=10000, help="Число симуляций.")
    p.add_argument("--seed", type=int, help="Сид для ГПСЧ.")
    p.add_argument("--heavy-tailed", action="store_true", help="Использовать t-распределение (df=5) вместо нормального.")

    # Прямой ввод параметров без файла
    p.add_argument("--wr_bb100", type=float, help="(cash) винрейт в bb/100.")
    p.add_argument("--sd_bb100", type=float, help="(cash) σ в bb/100.")
    p.add_argument("--roi_ev", type=float, help="(mtt) ожидаемый ROI в долях (0.1 = 10%).")
    p.add_argument("--sd_buyins", type=float, help="(mtt) σ результата в бай-инах за турнир.")
    p.add_argument("--hands", type=int, help="(cash) объём рук для оценки.")
    p.add_argument("--sessions", type=int, help="(cash) число сессий (если моделируем по сессиям).")
    p.add_argument("--samples", type=int, help="(mtt) число турниров (если моделируем по турнирам).")

    # Точность/вывод
    p.add_argument("--alpha", type=float, default=0.05, help="Уровень значимости для доверительного интервала.")
    p.add_argument("--plot", type=str, help="Путь к PNG-графику для сохранения.")
    p.add_argument("--out", type=str, help="CSV с финальными результатами симуляций.")
    p.add_argument("--json", dest="json_out", type=str, help="JSON-файл со сводкой.")
    p.add_argument("--quiet", action="store_true", help="Минимальный консольный вывод.")
    p.add_argument("--no-ansi", action="store_true", help="Отключить цветной вывод.")

    args = p.parse_args()
    return args


# ============================
# Утилиты парсинга величин
# ============================

_re_number = re.compile(r"^\s*([-+]?\d+(\.\d+)?)\s*\$?\s*$")  # "10" или "10$" → 10.0
_re_horizon = re.compile(r"^\s*(\d+)\s*([hts])\s*$", re.I)  # "5000h" / "50s" / "100t"
_re_threshold = re.compile(r"^\s*([-+]?\d+(\.\d+)?)\s*(bb|bi|\$)\s*$", re.I)


def parse_number_like(txt: Optional[str]) -> Optional[float]:
    if not txt:
        return None
    m = _re_number.match(txt)
    if not m:
        raise ValueError(f"Не удалось интерпретировать число: {txt}")
    return float(m.group(1))


def parse_horizon(txt: str) -> Tuple[int, str]:
    """
    Возвращает (число_шагов, тип_шага):
    - cash: 'h' → руки (конвертируется в шаги по 100 рук), 's' → сессии
    - mtt:  't' → турниры
    """
    m = _re_horizon.match(txt or "")
    if not m:
        raise ValueError("Неверный формат --horizon. Примеры: '5000h', '50s', '100t'.")
    count = int(m.group(1))
    kind = m.group(2).lower()
    return count, kind


def parse_threshold(txt: str) -> Tuple[float, str]:
    m = _re_threshold.match(txt or "")
    if not m:
        raise ValueError("Неверный формат --drawdown. Пример: '20bi', '1000bb', '500$'.")
    val = float(m.group(1))
    unit = m.group(3).lower()
    return val, unit


def z_value(alpha: float) -> float:
    """Z для нормального распределения (инверсия через аппрокс. Поляка)."""
    # Аппроксимация обратной функции Лапласа (Beasley-Springer/Moro) – достаточно точна для CI
    if not (0 < alpha < 1):
        raise ValueError("alpha должен быть в (0,1)")
    p = 1 - alpha / 2.0
    # Коэффициенты Моро:
    a = [2.50662823884,
         -18.61500062529,
         41.39119773534,
         -25.44106049637]
    b = [-8.47351093090,
         23.08336743743,
         -21.06224101826,
         3.13082909833]
    c = [0.3374754822726147,
         0.9761690190917186,
         0.1607979714918209,
         0.0276438810333863,
         0.0038405729373609,
         0.0003951896511919,
         0.0000321767881768,
         0.0000002888167364,
         0.0000003960315187]

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
        den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) + 1.0
        x = num / den
    else:
        r = p if y > 0 else 1 - p
        r = math.log(-math.log(r))
        x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
        if y < 0:
            x = -x
    return x


# ============================
# Загрузка и автоопределение режима
# ============================

def load_data(args: argparse.Namespace) -> Optional["pd.DataFrame"]:
    if args.input and args.json_in:
        raise SystemExit("Укажите либо --input (CSV), либо --json-in (JSON), но не оба сразу.")

    if args.input:
        if pd is None:
            raise SystemExit("Для чтения CSV нужен pandas. Установите: pip install pandas")
        if not os.path.exists(args.input):
            raise SystemExit(f"Файл не найден: {args.input}")
        df = pd.read_csv(args.input)
        if df.empty:
            raise SystemExit("CSV пуст.")
        return df

    if args.json_in:
        if pd is None:
            raise SystemExit("Для чтения JSON нужен pandas. Установите: pip install pandas")
        if not os.path.exists(args.json_in):
            raise SystemExit(f"Файл не найден: {args.json_in}")
        df = pd.read_json(args.json_in)
        if df.empty:
            raise SystemExit("JSON пуст.")
        return df

    # Нет файла — возможно прямой ввод параметров; вернем None
    return None


def detect_mode_and_units(df: Optional["pd.DataFrame"], args: argparse.Namespace) -> ModeContext:
    # Определяем BB и средний бай-ин (валюта → число)
    bb_value = parse_number_like(args.bb)
    avg_buyin = parse_number_like(args.avg_buyin)

    if args.mode:
        mode = args.mode
    else:
        # Простая эвристика по полям входного файла
        if df is not None:
            cols = set(c.lower() for c in df.columns)
            if {"buyin", "prize"} & cols:
                mode = "mtt"
            elif {"result"} & cols:
                mode = "cash"
            else:
                raise SystemExit("Не удалось автоопределить режим. Укажите --mode cash|mtt.")
        else:
            # Без файла — если заданы ROI или wr_bb100
            if args.roi_ev is not None or args.sd_buyins is not None or (args.samples is not None):
                mode = "mtt"
            elif args.wr_bb100 is not None or args.sd_bb100 is not None or (args.hands is not None) or (args.sessions is not None):
                mode = "cash"
            else:
                raise SystemExit("Укажите --mode cash|mtt (без входного файла не удалось определить режим).")

    # Горизонт
    horizon_steps = 0
    step_kind = ""

    if args.horizon:
        count, kind = parse_horizon(args.horizon)
        if mode == "cash":
            if kind == "h":
                # шаг — 100 рук
                horizon_steps = max(1, count // 100)  # количество шагов по 100 рук
                step_kind = "per100"
            elif kind == "s":
                horizon_steps = count
                step_kind = "sessions"
            else:
                raise SystemExit("Для cash используйте 'h' (руки) или 's' (сессии).")
        else:  # mtt
            if kind != "t":
                raise SystemExit("Для mtt используйте 't' (турниры).")
            horizon_steps = count
            step_kind = "tournaments"
    else:
        # Если нет horizon — попробуем из прямых параметров
        if mode == "cash":
            if args.hands:
                horizon_steps = max(1, args.hands // 100)
                step_kind = "per100"
            elif args.sessions:
                horizon_steps = args.sessions
                step_kind = "sessions"
            else:
                # По умолчанию 100 шагов (10k рук)
                horizon_steps = 100
                step_kind = "per100"
        else:
            if args.samples:
                horizon_steps = args.samples
                step_kind = "tournaments"
            else:
                horizon_steps = 100
                step_kind = "tournaments"

    ctx = ModeContext(
        mode=mode,
        units="bb" if mode == "cash" else "bi",
        bb_value=bb_value,
        avg_buyin=avg_buyin,
        horizon_steps=horizon_steps,
        step_kind=step_kind,
    )
    return ctx


# ============================
# Нормализация данных
# ============================

def normalize_cash(df: "pd.DataFrame", ctx: ModeContext) -> "pd.DataFrame":
    """
    Приводим результаты к bb и вычисляем wr_bb100 / sd_bb100, если есть 'hands'.
    Ожидаемые поля:
      - result (в bb ИЛИ в валюте)
      - hands (опционально)
      - bb (если result в валюте)
    """
    df = df.copy()

    # Попытка понять, в чем результат: если есть столбец 'bb' в датафрейме — считаем result в валюте.
    lower = {c.lower(): c for c in df.columns}
    has_bb_col = "bb" in lower
    col_bb = lower.get("bb")
    col_result = lower.get("result")
    col_hands = lower.get("hands")

    if col_result is None:
        raise SystemExit("Для cash ожидается колонка 'result' (итог сессии).")

    if has_bb_col:
        # result в валюте → переведем в bb по строковому bb
        if (df[col_bb] <= 0).any():
            raise SystemExit("Колонка 'bb' содержит неположительные значения.")
        df["result_bb"] = df[col_result] / df[col_bb]
    else:
        # result уже в bb
        df["result_bb"] = df[col_result]

    # hands опционально
    if col_hands and df[col_hands].notna().all():
        df["hands"] = df[col_hands].astype(float)
    else:
        df["hands"] = np.nan  # при отсутствии рук считаем по сессиям

    return df


def normalize_mtt(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Приводим результаты к бай-инам: result_bi = (prize - buyin)/buyin
    Ожидаемые поля: buyin, prize
    """
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}
    col_buyin = lower.get("buyin")
    col_prize = lower.get("prize")
    if col_buyin is None or col_prize is None:
        raise SystemExit("Для mtt ожидаются колонки 'buyin' и 'prize'.")

    if (df[col_buyin] <= 0).any():
        raise SystemExit("Колонка 'buyin' содержит неположительные значения.")

    df["result_bi"] = (df[col_prize] - df[col_buyin]) / df[col_buyin]
    return df


# ============================
# Базовые метрики
# ============================

def compute_basics_from_df(df: "pd.DataFrame", ctx: ModeContext, alpha: float) -> Basics:
    if ctx.mode == "cash":
        # Если есть руки — оценим wr/sd в bb/100 по агрегированию
        if df["hands"].notna().all():
            # Суммарный результат и суммарные руки
            total_result_bb = float(df["result_bb"].sum())
            total_hands = float(df["hands"].sum())
            if total_hands <= 0:
                raise SystemExit("hands должны быть положительными.")
            wr_bb100 = 100.0 * total_result_bb / total_hands

            # Оценка σ_bb100 по батчам (каждая строка — батч с количеством рук)
            # Преобразуем в результат на 100 рук: r_100 = 100 * result_bb / hands
            r_100 = 100.0 * df["result_bb"].values / df["hands"].values
            sd_bb100 = float(np.std(r_100, ddof=1))

            # Для общих mean/std используем результат в bb по сессиям (информативно)
            mean = float(df["result_bb"].mean())
            median = float(df["result_bb"].median())
            std = float(np.std(df["result_bb"].values, ddof=1))
            n = int(df.shape[0])

            # CI для винрейта wr_bb100 (по r_100)
            z = z_value(alpha)
            ci = (float(np.mean(r_100) - z * sd_bb100 / math.sqrt(n)),
                  float(np.mean(r_100) + z * sd_bb100 / math.sqrt(n)))

            return Basics(
                n_obs=n,
                mean=mean,
                median=median,
                std=std,
                cv=(std / abs(mean)) if mean != 0 else float("inf"),
                ci_low=ci[0],
                ci_high=ci[1],
                wr_bb100=wr_bb100,
                sd_bb100=sd_bb100,
            )

        else:
            # Нет hands → считаем по сессиям (грубее)
            r = df["result_bb"].values.astype(float)
            mean = float(np.mean(r))
            median = float(np.median(r))
            std = float(np.std(r, ddof=1))
            n = int(df.shape[0])
            z = z_value(alpha)
            ci = (mean - z * std / math.sqrt(n), mean + z * std / math.sqrt(n))
            if n < 30:
                print("⚠️  Внимание: нет 'hands' — σ и CI по сессиям могут быть шумными.", file=sys.stderr)
            return Basics(
                n_obs=n,
                mean=mean,
                median=median,
                std=std,
                cv=(std / abs(mean)) if mean != 0 else float("inf"),
                ci_low=ci[0],
                ci_high=ci[1],
                wr_bb100=None,
                sd_bb100=None,
            )

    # MTT
    r = df["result_bi"].values.astype(float)
    mean = float(np.mean(r))
    median = float(np.median(r))
    std = float(np.std(r, ddof=1))
    n = int(df.shape[0])
    z = z_value(alpha)
    ci = (mean - z * std / math.sqrt(n), mean + z * std / math.sqrt(n))

    return Basics(
        n_obs=n,
        mean=mean,
        median=median,
        std=std,
        cv=(std / abs(mean)) if mean != 0 else float("inf"),
        ci_low=ci[0],
        ci_high=ci[1],
        roi=mean,
        sd_bi=std,
    )


def compute_basics_from_params(ctx: ModeContext, args: argparse.Namespace, alpha: float) -> Basics:
    # Расчет baseline по прямым параметрам (без файла)
    if ctx.mode == "cash":
        if args.wr_bb100 is None or args.sd_bb100 is None:
            raise SystemExit("Для cash без файла укажите --wr_bb100 и --sd_bb100.")
        wr = float(args.wr_bb100)
        sd = float(args.sd_bb100)
        # Считаем фиктивный n для CI: возьмем n= max(30, steps) — просто для информативного CI
        n = max(30, ctx.horizon_steps)
        z = z_value(alpha)
        ci = (wr - z * sd / math.sqrt(n), wr + z * sd / math.sqrt(n))
        return Basics(
            n_obs=n,
            mean=wr, median=wr, std=sd, cv=(sd / abs(wr)) if wr != 0 else float("inf"),
            ci_low=ci[0], ci_high=ci[1],
            wr_bb100=wr, sd_bb100=sd
        )
    else:
        if args.roi_ev is None or args.sd_buyins is None:
            raise SystemExit("Для mtt без файла укажите --roi_ev и --sd_buyins.")
        roi = float(args.roi_ev)
        sd = float(args.sd_buyins)
        n = max(30, ctx.horizon_steps)
        z = z_value(alpha)
        ci = (roi - z * sd / math.sqrt(n), roi + z * sd / math.sqrt(n))
        return Basics(
            n_obs=n,
            mean=roi, median=roi, std=sd, cv=(sd / abs(roi)) if roi != 0 else float("inf"),
            ci_low=ci[0], ci_high=ci[1],
            roi=roi, sd_bi=sd
        )


# ============================
# Риск-метрики (приближение)
# ============================

def prob_down_ge_threshold(ctx: ModeContext, basics: Basics, threshold_val: float, threshold_unit: str) -> float:
    """
    Вероятность, что итоговый результат за горизонт ≤ порога (отрицательного).
    """
    if ctx.mode == "cash":
        # Работает в единицах bb.
        if threshold_unit == "$":
            if ctx.bb_value is None:
                raise SystemExit("Для порога в валюте укажите --bb (размер большого блайнда).")
            T_bb = threshold_val / ctx.bb_value
        elif threshold_unit == "bb":
            T_bb = threshold_val
        elif threshold_unit == "bi":
            # 1 би = 100 bb в кэше (условное допущение)
            T_bb = threshold_val * 100.0
        else:
            raise SystemExit("Неверная единица порога для cash.")

        # Ожидание и дисперсия суммы за horizon_steps шагов
        # шаг = 100 рук ИЛИ сессия. Для сессий нет wr/sd → используем оценку по mean/std на сессию.
        if ctx.step_kind == "per100" and basics.wr_bb100 is not None and basics.sd_bb100 is not None:
            mu = basics.wr_bb100
            sd = basics.sd_bb100
            mean_sum = ctx.horizon_steps * mu
            var_sum = ctx.horizon_steps * (sd ** 2)
        else:
            # по сессиям: mean/std в bb на сессию
            mean_sum = ctx.horizon_steps * basics.mean
            var_sum = ctx.horizon_steps * (basics.std ** 2)

        z = (T_bb - mean_sum) / math.sqrt(max(1e-12, var_sum))
        # P(S ≤ T) = Φ(z). Используем норм. приближение через erf:
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # mtt, единицы — bi
    if threshold_unit == "$":
        if ctx.avg_buyin is None:
            raise SystemExit("Для порога в валюте укажите --avg_buyin (средний бай-ин).")
        T_bi = threshold_val / ctx.avg_buyin
    elif threshold_unit == "bb":
        # не применимо для MTT; поддержим на всякий случай: 1 би ~ 100 bb на кэше невозможно связать
        raise SystemExit("Для mtt используйте 'bi' или '$' для порога.")
    elif threshold_unit == "bi":
        T_bi = threshold_val
    else:
        raise SystemExit("Неверная единица порога для mtt.")

    mu = basics.roi if basics.roi is not None else 0.0
    sd = basics.sd_bi if basics.sd_bi is not None else 0.0

    mean_sum = ctx.horizon_steps * mu
    var_sum = ctx.horizon_steps * (sd ** 2)
    z = (T_bi - mean_sum) / math.sqrt(max(1e-12, var_sum))
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def risk_of_ruin_approx(ctx: ModeContext, basics: Basics, bankroll: Optional[float]) -> Optional[float]:
    """
    Приближенная формула RoR ~ exp(-2 * μ * BR / σ^2) при μ>0.
    Единицы:
      - cash: bb (если есть bb_value и пользователь дал BR в $ — конвертируем)
      - mtt: bi (если пользователь дал BR в $ — конвертируем через avg_buyin)
    """
    if bankroll is None or bankroll <= 0:
        return None

    if ctx.mode == "cash":
        # Переведём BR в bb при необходимости
        if ctx.bb_value and (ctx.units == "bb" or True):  # банкролл пользователь мог вводить в $
            BR_bb = bankroll / ctx.bb_value if ctx.bb_value else bankroll
        else:
            BR_bb = bankroll
        if ctx.step_kind == "per100" and basics.wr_bb100 and basics.sd_bb100:
            mu = basics.wr_bb100
            sigma2 = basics.sd_bb100 ** 2
        else:
            # шаг по сессии: му и σ на сессию
            mu = basics.mean
            sigma2 = basics.std ** 2

        if mu <= 0:
            return 1.0
        return float(math.exp(-2.0 * mu * BR_bb / max(1e-12, sigma2)))

    # mtt
    if ctx.avg_buyin:
        BR_bi = bankroll / ctx.avg_buyin
    else:
        # Если не указали средний бай-ин — считаем, что банкролл введён уже в би
        BR_bi = bankroll

    mu = basics.roi if basics.roi is not None else 0.0
    sigma2 = (basics.sd_bi if basics.sd_bi is not None else 0.0) ** 2
    if mu <= 0:
        return 1.0
    return float(math.exp(-2.0 * mu * BR_bi / max(1e-12, sigma2)))


# ============================
# Монте-Карло симуляции
# ============================

def simulate(ctx: ModeContext,
             basics: Basics,
             nsims: int,
             seed: Optional[int],
             heavy_tailed: bool,
             target: Optional[float],
             drawdown: Optional[Tuple[float, str]],
             sample_paths: int = 20) -> SimResult:
    """
    Возвращает массив финальных результатов, флаги банкротств, квантиля и т.д.
    Единицы:
      - cash: bb (для per100/sessions)
      - mtt: bi
    """
    if seed is not None:
        np.random.seed(seed)

    steps = int(ctx.horizon_steps)
    if steps <= 0:
        raise SystemExit("Горизонт моделирования должен быть положительным.")

    # Генерация приращений
    if ctx.mode == "cash":
        if ctx.step_kind == "per100" and basics.wr_bb100 is not None and basics.sd_bb100 is not None:
            mu = basics.wr_bb100
            sd = basics.sd_bb100
        else:
            # по сессиям
            mu = basics.mean
            sd = basics.std
    else:
        mu = basics.roi if basics.roi is not None else 0.0
        sd = basics.sd_bi if basics.sd_bi is not None else 0.0

    if heavy_tailed:
        # t-распределение с df=5, нормируем под нужные mu/sd
        df = 5.0
        # Стандартное отклонение t(df) = sqrt(df/(df-2)) для df>2
        std_t = math.sqrt(df / (df - 2.0))
        def draw(size):  # noqa: E306
            return mu + sd * (np.random.standard_t(df, size=size) / std_t)
    else:
        def draw(size):  # noqa: E306
            return np.random.normal(loc=mu, scale=sd, size=size)

    # Симуляции
    X = draw((nsims, steps))
    paths = np.cumsum(X, axis=1)
    final = paths[:, -1]

    # Просадки и банкротства
    # Банкротство: BR + path < 0 для любой точки; BR берём из CLI target? — нет, нужен отдельный ввод.
    bankruptcies = None
    BR = None

    # Попробуем собрать банкролл из аргументов напрямую: пользователь мог ввести разными единицами,
    # здесь считаем, что он ввёл в текущих единицах модели (bb для cash, bi для mtt).
    # Если пользователь указывал в валюте и дал bb/avg_buyin — логично было конвертировать до вызова simulate.
    # Мы читаем из ENV-переменной-посредника, которую set'нем в main() (не самый элегантный, но компактно).
    _br_env = os.environ.get("VARIANCE_BR_MODEL_UNITS")
    if _br_env:
        try:
            BR = float(_br_env)
        except Exception:
            BR = None

    if BR is not None and BR > 0:
        balance = BR + paths
        bankruptcies = (balance <= 0).any(axis=1)

    # Просадка (максимальная)
    running_max = np.maximum.accumulate(np.concatenate([np.zeros((nsims, 1)), paths], axis=1), axis=1)[:, 1:]
    drawdowns = running_max - paths
    max_dd = np.max(drawdowns, axis=1)

    # Квантиль
    p10 = float(np.percentile(final, 10))
    p50 = float(np.percentile(final, 50))
    p90 = float(np.percentile(final, 90))

    # Вероятность достижения цели и просадки >= threshold
    p_hit_target = None
    if target is not None:
        p_hit_target = float(np.mean(final >= target))

    p_hit_drawdown = None
    if drawdown is not None:
        thr_val, thr_unit = drawdown
        # drawdowns в единицах модели уже (bb/bi). Если порог в $, конвертация выполняется заранее в main.
        p_hit_drawdown = float(np.mean(max_dd >= thr_val))

    risk_mc = float(np.mean(bankruptcies)) if bankruptcies is not None else 0.0

    # Примеры траекторий для графика
    k = min(sample_paths, nsims)
    sample_trajectories = paths[:k, :]

    return SimResult(
        final_results=final,
        drawdowns=max_dd,
        bankruptcies=bankruptcies,
        p10=p10, p50=p50, p90=p90,
        risk_of_ruin_mc=risk_mc,
        p_hit_target=p_hit_target,
        p_hit_drawdown=p_hit_drawdown,
        sample_trajectories=sample_trajectories,
    )


# ============================
# Рендер/сохранение
# ============================

def render_report(ctx: ModeContext,
                  basics: Basics,
                  risks: RiskMetrics,
                  sim: Optional[SimResult],
                  args: argparse.Namespace) -> None:
    if args.quiet:
        return

    def fmt(x: Optional[float]) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{x:.4f}"

    print("\n=== СВОДКА ===")
    print(f"Режим: {ctx.mode.upper()}, единицы: {ctx.units}")
    print(f"Выборка: n={basics.n_obs}")
    if ctx.mode == "cash":
        if basics.wr_bb100 is not None:
            print(f"wr_bb100: {basics.wr_bb100:.3f} | sd_bb100: {basics.sd_bb100:.3f} | CI({1-args.alpha:.0%}): [{basics.ci_low:.3f}, {basics.ci_high:.3f}]")
        else:
            print(f"mean(bb/сессия): {basics.mean:.3f} | std: {basics.std:.3f} | CI: [{basics.ci_low:.3f}, {basics.ci_high:.3f}]")
    else:
        print(f"ROI (в би): {basics.roi:.4f} | σ (в би): {basics.sd_bi:.4f} | CI: [{basics.ci_low:.4f}, {basics.ci_high:.4f}]")

    print(f"median: {basics.median:.4f} | std: {basics.std:.4f} | CV: {basics.cv:.4f}")

    print("\n=== РИСК-МЕТРИКИ (приближение) ===")
    print(f"P(просадка ≥ threshold за горизонт): {fmt(risks.p_down_ge_threshold)}")
    print(f"Risk of Ruin (аппроксимация): {fmt(risks.risk_of_ruin_approx)}")
    print(f"P(достичь цели): {fmt(risks.p_target)}")
    print(f"P(уйти в минус): {fmt(risks.p_loss)}")

    if sim is not None:
        print("\n=== МОНТЕ-КАРЛО ===")
        print(f"nsims: {args.nsims} | распределение финала: p10={sim.p10:.3f}, p50={sim.p50:.3f}, p90={sim.p90:.3f}")
        print(f"Risk of Ruin (MC): {sim.risk_of_ruin_mc:.4f}")
        if sim.p_hit_target is not None:
            print(f"P(final ≥ target): {sim.p_hit_target:.4f}")
        if sim.p_hit_drawdown is not None:
            print(f"P(max drawdown ≥ threshold): {sim.p_hit_drawdown:.4f}")


def save_artifacts(sim: Optional[SimResult],
                   basics: Basics,
                   risks: RiskMetrics,
                   ctx: ModeContext,
                   args: argparse.Namespace) -> None:
    # CSV финалов
    if sim is not None and args.out:
        if pd is None:
            print("⚠️  Не могу сохранить CSV (нет pandas). Установите pandas.", file=sys.stderr)
        else:
            out_df = pd.DataFrame({
                "final_result": sim.final_results
            })
            out_df.to_csv(args.out, index=False)

    # JSON сводки
    if args.json_out:
        payload: Dict[str, Any] = {
            "mode": ctx.mode,
            "units": ctx.units,
            "basics": {
                "n_obs": basics.n_obs,
                "mean": basics.mean,
                "median": basics.median,
                "std": basics.std,
                "cv": basics.cv,
                "ci_low": basics.ci_low,
                "ci_high": basics.ci_high,
                "wr_bb100": basics.wr_bb100,
                "sd_bb100": basics.sd_bb100,
                "roi": basics.roi,
                "sd_bi": basics.sd_bi,
            },
            "risks": {
                "p_down_ge_threshold": risks.p_down_ge_threshold,
                "risk_of_ruin_approx": risks.risk_of_ruin_approx,
                "p_target": risks.p_target,
                "p_loss": risks.p_loss,
            }
        }
        if sim is not None:
            payload["simulation"] = {
                "p10": sim.p10,
                "p50": sim.p50,
                "p90": sim.p90,
                "risk_of_ruin_mc": sim.risk_of_ruin_mc,
                "p_hit_target": sim.p_hit_target,
                "p_hit_drawdown": sim.p_hit_drawdown,
            }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_figure(sim: SimResult, ctx: ModeContext, args: argparse.Namespace) -> None:
    if not args.plot:
        return
    if plt is None:
        print("⚠️  Не могу построить график (нет matplotlib). Установите matplotlib.", file=sys.stderr)
        return

    plt.figure(figsize=(10, 6))
    # Траектории
    if sim.sample_trajectories is not None:
        for i in range(sim.sample_trajectories.shape[0]):
            plt.plot(sim.sample_trajectories[i], alpha=0.4)
    plt.title(f"Траектории {'банкролла' if ctx.mode=='cash' else 'результата'} ({ctx.mode.upper()}, {ctx.horizon_steps} шагов)")
    plt.xlabel("Шаг")
    plt.ylabel(f"Результат ({ctx.units})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=140)
    plt.close()


# ============================
# main
# ============================

def main():
    args = parse_args()

    # Загрузка данных
    df = load_data(args)

    # Контекст и единицы
    ctx = detect_mode_and_units(df, args)

    # Предупреждения по зависимостям
    if (args.input or args.json_in or args.out) and pd is None:
        print("⚠️  Для работы с CSV/JSON требуется pandas. Установите: pip install pandas", file=sys.stderr)

    # Нормализация и базовые метрики
    if df is not None:
        if pd is None:
            raise SystemExit("Нельзя читать/нормализовать данные без pandas.")
        if df.shape[0] < 5:
            raise SystemExit("Недостаточно наблюдений (<5).")
        if ctx.mode == "cash":
            df_norm = normalize_cash(df, ctx)
        else:
            df_norm = normalize_mtt(df)

        basics = compute_basics_from_df(df_norm, ctx, alpha=args.alpha)
    else:
        basics = compute_basics_from_params(ctx, args, alpha=args.alpha)

    # Разбор drawdown порога и конвертации
    dd_val = None
    dd_unit = None
    if args.drawdown:
        val, unit = parse_threshold(args.drawdown)
        if ctx.mode == "cash":
            # приведение к bb для вероятности (внутри prob_down_ge_threshold преобразуем автоматически)
            dd_val, dd_unit = val, unit
        else:
            # mtt — bi/$ валиден
            dd_val, dd_unit = val, unit

    # Цель в единицах модели (bb/bi). Если пользователю удобно $, конвертируем.
    target_model = args.target
    if target_model is not None:
        if ctx.mode == "cash" and args.currency and ctx.bb_value:
            # Если target задан по умолчанию в $, конвертировать: но мы не знаем, что ввёл пользователь.
            # Предполагаем: он ввёл в единицах модели. Не преобразуем автоматически, чтобы не гадать.
            pass
        if ctx.mode == "mtt" and ctx.avg_buyin and args.currency:
            pass

    # Оценка вероятности даунстрика (аппроксимация)
    p_dd = None
    if dd_val is not None and dd_unit is not None:
        p_dd = prob_down_ge_threshold(ctx, basics, dd_val, dd_unit)

    # Risk of Ruin (аппроксимация)
    bankroll_cli: Optional[float] = None
    # Пользователь мог записать банкролл в тех же единицах, что и порог.
    # В ТЗ банкролл вводится отдельным флагом --bankroll (необязателен в коде для краткости).
    # Добавим поддержку: прочитаем из дополнительных позиционных env-параметров, если присутствуют:
    # Для простоты примем формат как число плюс постфикс единиц (bb/bi/$), аналогично --drawdown.
    br_env = os.environ.get("VARIANCE_BANKROLL")
    if br_env:
        try:
            br_val, br_unit = parse_threshold(br_env)
            if ctx.mode == "cash":
                if br_unit == "$":
                    if ctx.bb_value is None:
                        raise SystemExit("Для BR в валюте укажите --bb.")
                    bankroll_cli = br_val / ctx.bb_value  # приведем к bb для RoR
                elif br_unit == "bb":
                    bankroll_cli = br_val
                elif br_unit == "bi":
                    bankroll_cli = br_val * 100.0  # 1 би = 100 bb (условность)
            else:
                if br_unit == "$":
                    if ctx.avg_buyin is None:
                        raise SystemExit("Для BR в валюте укажите --avg_buyin.")
                    bankroll_cli = br_val / ctx.avg_buyin  # приведем к bi
                elif br_unit == "bi":
                    bankroll_cli = br_val
                else:
                    raise SystemExit("Для mtt BR задавайте в 'bi' или '$'.")
        except Exception as e:
            print(f"⚠️  Не удалось разобрать VARIANCE_BANKROLL: {e}", file=sys.stderr)

    ror_approx = risk_of_ruin_approx(ctx, basics, bankroll_cli)

    # P(loss) и P(target) в приближении норм. суммы
    p_loss = None
    p_target = None
    if ctx.mode == "cash":
        if ctx.step_kind == "per100" and basics.wr_bb100 is not None and basics.sd_bb100 is not None:
            mu = basics.wr_bb100
            sd = basics.sd_bb100
        else:
            mu = basics.mean
            sd = basics.std
        mean_sum = ctx.horizon_steps * mu
        var_sum = ctx.horizon_steps * (sd ** 2)
        p_loss = 0.5 * (1 + math.erf((0.0 - mean_sum) / math.sqrt(max(1e-12, var_sum)) / math.sqrt(2)))
        if target_model is not None:
            zt = (target_model - mean_sum) / math.sqrt(max(1e-12, var_sum))
            p_target = 1.0 - (0.5 * (1 + math.erf(zt / math.sqrt(2))))
    else:
        mu = basics.roi if basics.roi is not None else 0.0
        sd = basics.sd_bi if basics.sd_bi is not None else 0.0
        mean_sum = ctx.horizon_steps * mu
        var_sum = ctx.horizon_steps * (sd ** 2)
        p_loss = 0.5 * (1 + math.erf((0.0 - mean_sum) / math.sqrt(max(1e-12, var_sum)) / math.sqrt(2)))
        if target_model is not None:
            zt = (target_model - mean_sum) / math.sqrt(max(1e-12, var_sum))
            p_target = 1.0 - (0.5 * (1 + math.erf(zt / math.sqrt(2))))

    risks = RiskMetrics(
        p_down_ge_threshold=p_dd,
        risk_of_ruin_approx=ror_approx,
        p_target=p_target,
        p_loss=p_loss,
    )

    # Монте-Карло
    # Для корректного учета банкролла в MC прогонах нам нужно знать BR в единицах модели — передадим через ENV:
    if bankroll_cli is not None:
        os.environ["VARIANCE_BR_MODEL_UNITS"] = str(bankroll_cli)
    else:
        os.environ.pop("VARIANCE_BR_MODEL_UNITS", None)

    sim: Optional[SimResult] = simulate(
        ctx=ctx,
        basics=basics,
        nsims=args.nsims,
        seed=args.seed,
        heavy_tailed=args.heavy_tailed,
        target=target_model,
        drawdown=(dd_val, dd_unit) if (dd_val is not None and dd_unit is not None) else None,
        sample_paths=20
    )

    # Вывод и артефакты
    render_report(ctx, basics, risks, sim, args)
    save_artifacts(sim, basics, risks, ctx, args)
    if sim is not None:
        plot_figure(sim, ctx, args)

    # Дружелюбный финал
    if not args.quiet:
        print("\nГотово ✔")


if __name__ == "__main__":
    main()
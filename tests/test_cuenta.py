"""Perfil de cuenta: la demo por defecto, y una cuenta real solo si se declara como tal."""
from pathlib import Path

import pytest

import cuenta


def test_sin_configuracion_es_la_demo_de_siempre():
    perfil = cuenta.desde_entorno({})
    assert perfil is cuenta.DEMO and not perfil.real and not perfil.exigir_ventaja
    assert cuenta.CUENTA is cuenta.DEMO  # conftest deja el entorno de los tests sin cuenta configurada


def test_otra_demo_usa_la_terminal_por_defecto_y_no_exige_ventaja():
    perfil = cuenta.desde_entorno({"EXNESS_LOGIN": "123", "EXNESS_SERVER": "Exness-MT5Trial7"})
    assert perfil == cuenta.Cuenta(123, "Exness-MT5Trial7", cuenta.DEMO.terminal, real=False, exigir_ventaja=False)


def test_una_cuenta_real_exige_ventaja_por_defecto():
    perfil = cuenta.desde_entorno({"EXNESS_LOGIN": "123", "EXNESS_SERVER": "Exness-MT5Real8", "EXNESS_CUENTA_REAL": "1",
                                   "EXNESS_TERMINAL": r"D:\mt5\terminal64.exe"})
    assert (perfil.real, perfil.exigir_ventaja, perfil.terminal, perfil.tipo) == (True, True, r"D:\mt5\terminal64.exe", "real")


def test_la_exigencia_de_ventaja_se_puede_cambiar_a_proposito():
    base = {"EXNESS_LOGIN": "123", "EXNESS_SERVER": "s"}
    assert not cuenta.desde_entorno({**base, "EXNESS_CUENTA_REAL": "1", "EXNESS_EXIGIR_VENTAJA": "0"}).exigir_ventaja
    assert cuenta.desde_entorno({**base, "EXNESS_EXIGIR_VENTAJA": "1"}).exigir_ventaja


@pytest.mark.parametrize("entorno,mensaje", [
    ({"EXNESS_LOGIN": "123"}, "van juntos"), ({"EXNESS_SERVER": "s"}, "van juntos"),
    ({"EXNESS_CUENTA_REAL": "1"}, "exige indicar"), ({"EXNESS_LOGIN": "abc", "EXNESS_SERVER": "s"}, "número de cuenta"),
    ({"EXNESS_LOGIN": "123", "EXNESS_SERVER": "s", "EXNESS_CUENTA_REAL": "quizá"}, "usa 1 o 0"),
])
def test_una_configuracion_a_medias_no_arranca(entorno, mensaje):
    with pytest.raises(SystemExit, match=mensaje):
        cuenta.desde_entorno(entorno)


def test_cada_cuenta_tiene_su_carpeta_y_la_demo_conserva_la_suya():
    base = Path("logs") / "exness"
    assert cuenta.DEMO.carpeta(base) == base
    assert cuenta.Cuenta(123, "s", "t").carpeta(base) == base / "123"

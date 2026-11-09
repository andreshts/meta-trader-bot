"""La hora común de los modelos y la pausa diaria de los activos que cierran una hora."""
from datetime import datetime, timezone

import numpy as np
import pytest

import activos
import train_exness


def ts(*fecha) -> int:
    return int(datetime(*fecha, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("fecha,horas", [((2026, 1, 15, 12), 2), ((2026, 7, 15, 12), 3), ((2026, 11, 9, 12), 2)])
def test_la_hora_del_modelo_sigue_el_horario_de_eeuu(fecha, horas):
    assert activos.hora_modelo([ts(*fecha)])[0] == ts(*fecha) + horas * 3600


@pytest.mark.parametrize("fecha", [(2026, 1, 15, 22, 0), (2026, 7, 15, 21, 0)])
def test_el_cierre_de_nueva_york_cae_a_medianoche_todo_el_anio(fecha):
    assert activos.hora_modelo([ts(*fecha)])[0] % 86400 == 0


@pytest.mark.parametrize("anterior,siguiente,esperado", [
    (ts(2026, 7, 15, 20, 45), ts(2026, 7, 15, 22, 0), True),     # pausa de verano: 21:00 a 22:00 UTC
    (ts(2026, 1, 15, 21, 45), ts(2026, 1, 15, 23, 0), True),     # pausa de invierno: 22:00 a 23:00 UTC
    (ts(2026, 7, 15, 14, 0), ts(2026, 7, 15, 15, 15), False),    # hueco a media sesión: datos incompletos
    (ts(2026, 7, 15, 20, 45), ts(2026, 7, 16, 2, 0), False),     # demasiado largo para ser la pausa
])
def test_solo_se_admite_la_pausa_diaria(anterior, siguiente, esperado):
    assert activos.es_cierre_diario(anterior, siguiente) is esperado


def test_el_umbral_inicial_da_la_frecuencia_pedida():
    proba = np.linspace(0.1, 0.4, 4600)  # 10 semanas de velas
    umbral = train_exness.umbral_por_frecuencia(proba, semanas=10)
    assert umbral == 0.39
    assert 50 <= np.count_nonzero(proba >= umbral) <= 160

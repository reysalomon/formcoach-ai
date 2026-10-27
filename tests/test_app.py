"""Проверка интерфейса без браузера (Streamlit AppTest)."""

from streamlit.testing.v1 import AppTest

DEMO = "Демо (синтетические данные)"


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_starts_without_video():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert at.info  # просит загрузить видео


def test_demo_mode_finds_errors():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.sidebar.radio[0].set_value(DEMO).run()
    assert not at.exception
    # по умолчанию: хорошее, неглубокое, хорошее, наклон, хорошее
    m = metrics(at)
    assert (m["Повторений"], m["Правильных"], m["С ошибками"]) == ("5", "3", "2")


def test_relaxed_depth_rule_changes_result():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.sidebar.radio[0].set_value(DEMO).run()
    at.sidebar.slider[3].set_value(140).run()  # порог глубины 140° — неглубокий присед теперь ок
    assert metrics(at)["Правильных"] == "4"

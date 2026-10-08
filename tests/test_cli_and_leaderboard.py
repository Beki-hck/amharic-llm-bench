import json

from evalforge.models import FunctionModel
from evalforge.runner import run_suite

from amharic_bench import leaderboard
from amharic_bench.cli import main
from amharic_bench.suites import SUITES, load


def oracle(suite):
    """A fake model that returns each item's gold answer."""
    gold = {it.prompt: it.expected for it in suite.items}

    def fn(prompt):
        exp = gold[prompt]
        return f"መልስ: {exp}" if isinstance(exp, str) else exp[0]

    return FunctionModel(fn, label="oracle")


def test_oracle_tops_the_leaderboard_with_full_marks():
    runs = []
    for name in SUITES:
        suite = load(name)
        runs.append(run_suite(suite, oracle(suite)).to_dict())
        runs.append(run_suite(suite, FunctionModel(lambda p: "I don't speak Amharic.", label="english-only")).to_dict())
    rows = leaderboard.build(runs)
    assert [r["model"] for r in rows] == ["oracle", "english-only"]
    assert rows[0]["average"] == 100.0 and rows[0]["script_rate"] == 100.0
    assert rows[1]["script_rate"] == 0.0
    md = leaderboard.markdown(rows)
    assert "| oracle | **100.0** |" in md


def test_bootstrap_ci_brackets_the_mean_and_is_repeatable():
    vals = [100.0] * 15 + [0.0] * 10
    lo, hi = leaderboard.bootstrap_ci(vals)
    assert lo < 60 < hi and (lo, hi) == leaderboard.bootstrap_ci(vals)
    assert leaderboard.bootstrap_ci([100.0] * 5) == (100.0, 100.0)


def test_cli_run_writes_results_and_leaderboard(tmp_path, capsys):
    assert main(["run", "-m", "echo:test", "-n", "2", "-o", str(tmp_path)]) == 0
    files = {p.name for p in tmp_path.iterdir()}
    assert {"leaderboard.html", "leaderboard.md", "leaderboard.json", "details.html"} <= files
    assert len([f for f in files if "__" in f]) == len(SUITES)
    rows = json.loads((tmp_path / "leaderboard.json").read_text(encoding="utf-8"))
    assert rows[0]["model"] == "echo:test"
    assert "Leaderboard:" in capsys.readouterr().out


def test_cli_list_and_normalize(capsys):
    assert main(["list"]) == 0 and "am_qa" in capsys.readouterr().out
    assert main(["normalize", "ሠላም"]) == 0 and capsys.readouterr().out.strip() == "ሰላም"


def test_ollama_answers_are_capped():
    from amharic_bench.models import CappedOllamaModel, get_model

    m = get_model("ollama:llama3.2:3b", max_tokens=64)
    assert isinstance(m, CappedOllamaModel) and m.max_tokens == 64 and m.name == "ollama:llama3.2:3b"
    assert get_model("echo:x").name == "echo:x"


def test_rescore_uses_current_answer_key(tmp_path):
    suite = load("am_qa")
    run = run_suite(suite, FunctionModel(lambda p: "ደሴ", label="m"), limit=1)
    path = run.save(tmp_path / "am_qa__m.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["results"][0].update(passed=False, value=0.0, detail="stale")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert main(["rescore", str(path), "-o", str(tmp_path)]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["passed"] == 1


def test_drift_is_counted_separately_from_english():
    from amharic_bench.normalize import ethiopic_ratio, foreign_script_letters

    assert foreign_script_letters("请发烧时大量饮水。") == 8 and ethiopic_ratio("请发烧时大量饮水。") == 0.0
    assert foreign_script_letters("ሰላም hello café") == 0
    assert foreign_script_letters("300 ኪሎ ግራም половине") == 8

"""Tests for evaluation runner module."""

import pytest
import tempfile
import os

from src.evaluation.runner import (
    EvaluationConfig, EvaluationRunner, EvaluationResult,
    quick_eval, full_eval
)
from src.game.opponents import RandomPlayer, CallingStation
from src.evaluation.analyzer import PerformanceMetrics


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_default_config(self):
        config = EvaluationConfig()

        assert config.num_hands == 1000
        assert config.opponent_types == ["fish", "tag"]
        assert config.big_blind == 1.0
        assert config.starting_stack == 100.0
        assert config.capture_traces is True

    def test_custom_config(self):
        config = EvaluationConfig(
            num_hands=500,
            opponent_types=["nit", "lag"],
            big_blind=2.0,
            starting_stack=50.0,
            capture_traces=False
        )

        assert config.num_hands == 500
        assert config.opponent_types == ["nit", "lag"]
        assert config.big_blind == 2.0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self):
        config = EvaluationConfig(num_hands=100)
        metrics = PerformanceMetrics(
            total_hands=100,
            total_profit_bb=25.0,
            bb_per_100=25.0
        )

        result = EvaluationResult(
            session_id="test_001",
            config=config,
            metrics=metrics,
            total_hands=100,
            total_profit_bb=25.0,
            duration_seconds=5.0
        )

        assert result.session_id == "test_001"
        assert result.total_hands == 100
        assert result.total_profit_bb == 25.0

    def test_result_str_format(self):
        config = EvaluationConfig(num_hands=100, opponent_types=["fish"])
        metrics = PerformanceMetrics(total_hands=100, total_profit_bb=10.0)

        result = EvaluationResult(
            session_id="test_002",
            config=config,
            metrics=metrics,
            total_hands=100,
            total_profit_bb=10.0,
            duration_seconds=3.0
        )

        output = str(result)
        assert "EVALUATION RESULT" in output
        assert "test_002" in output
        assert "100" in output


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    @pytest.fixture
    def simple_hero(self):
        return RandomPlayer("hero")

    @pytest.fixture
    def simple_config(self):
        return EvaluationConfig(
            num_hands=20,
            opponent_types=["random"],
            capture_traces=False,
            verbose=False
        )

    def test_runner_creation(self, simple_hero, simple_config):
        runner = EvaluationRunner(simple_hero, simple_config)

        assert runner.hero == simple_hero
        assert runner.config == simple_config
        assert runner.session_id.startswith("eval_")

    def test_run_returns_result(self, simple_hero, simple_config):
        runner = EvaluationRunner(simple_hero, simple_config)
        result = runner.run()

        assert isinstance(result, EvaluationResult)
        assert result.total_hands == 20

    def test_run_captures_profit(self, simple_hero, simple_config):
        runner = EvaluationRunner(simple_hero, simple_config)
        result = runner.run()

        # Profit should be recorded (could be positive, negative, or zero)
        assert isinstance(result.total_profit_bb, float)

    def test_run_tracks_duration(self, simple_hero, simple_config):
        runner = EvaluationRunner(simple_hero, simple_config)
        result = runner.run()

        assert result.duration_seconds > 0

    def test_run_with_multiple_opponents(self, simple_hero):
        config = EvaluationConfig(
            num_hands=20,
            opponent_types=["random", "fish"],
            capture_traces=False,
            verbose=False
        )

        runner = EvaluationRunner(simple_hero, config)
        result = runner.run()

        assert result.total_hands == 20

    def test_get_report_without_traces(self, simple_hero, simple_config):
        runner = EvaluationRunner(simple_hero, simple_config)
        runner.run()

        report = runner.get_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_get_report_before_run(self, simple_hero, simple_config):
        runner = EvaluationRunner(simple_hero, simple_config)
        report = runner.get_report()

        assert "No evaluation run yet" in report


class TestEvaluationRunnerWithTraces:
    """Tests for EvaluationRunner with trace capture enabled."""

    @pytest.fixture
    def hero(self):
        return RandomPlayer("hero")

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_run_with_traces(self, hero, temp_db):
        config = EvaluationConfig(
            num_hands=10,
            opponent_types=["random"],
            capture_traces=True,
            db_path=temp_db,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        result = runner.run()

        assert result.session_trace is not None
        assert len(result.session_trace.hands) == 10

    def test_analyzer_available_with_traces(self, hero, temp_db):
        config = EvaluationConfig(
            num_hands=10,
            opponent_types=["random"],
            capture_traces=True,
            db_path=temp_db,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        runner.run()

        assert runner.analyzer is not None
        report = runner.get_report()
        assert "POKER BOT EVALUATION REPORT" in report

    def test_export_traces(self, hero, temp_db):
        config = EvaluationConfig(
            num_hands=5,
            opponent_types=["random"],
            capture_traces=True,
            db_path=temp_db,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        runner.run()

        # Export to JSON
        json_path = temp_db.replace(".db", ".json")
        try:
            runner.export_traces(json_path)
            assert os.path.exists(json_path)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class TestQuickEval:
    """Tests for quick_eval helper function."""

    def test_quick_eval_runs(self):
        hero = RandomPlayer("hero")
        result = quick_eval(hero, opponent_type="random", num_hands=10, verbose=False)

        assert isinstance(result, EvaluationResult)
        assert result.total_hands == 10

    def test_quick_eval_no_traces(self):
        hero = RandomPlayer("hero")
        result = quick_eval(hero, opponent_type="random", num_hands=5, verbose=False)

        # Quick eval should not capture traces
        assert result.session_trace is None


class TestFullEval:
    """Tests for full_eval helper function."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_full_eval_runs(self, temp_db, capsys):
        hero = RandomPlayer("hero")
        result = full_eval(
            hero,
            opponent_types=["random"],
            num_hands=10,
            db_path=temp_db
        )

        assert isinstance(result, EvaluationResult)
        assert result.session_trace is not None

        # Should print report
        captured = capsys.readouterr()
        assert "POKER BOT EVALUATION REPORT" in captured.out


class TestEvaluationEdgeCases:
    """Edge case tests for evaluation."""

    def test_single_hand(self):
        hero = RandomPlayer("hero")
        config = EvaluationConfig(
            num_hands=1,
            opponent_types=["random"],
            capture_traces=False,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        result = runner.run()

        assert result.total_hands == 1

    def test_calling_station_hero(self):
        """Test with CallingStation as hero."""
        hero = CallingStation("hero")
        config = EvaluationConfig(
            num_hands=10,
            opponent_types=["random"],
            capture_traces=False,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        result = runner.run()

        assert result.total_hands == 10


class TestEvaluationMetrics:
    """Tests for metrics computation in evaluation."""

    def test_metrics_computed(self):
        hero = RandomPlayer("hero")
        config = EvaluationConfig(
            num_hands=50,
            opponent_types=["random"],
            capture_traces=False,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        result = runner.run()

        assert result.metrics is not None
        assert result.metrics.total_hands == 50

    def test_bb_per_100_computed(self):
        hero = RandomPlayer("hero")
        config = EvaluationConfig(
            num_hands=100,
            opponent_types=["random"],
            capture_traces=False,
            verbose=False
        )

        runner = EvaluationRunner(hero, config)
        result = runner.run()

        # BB/100 should be computed
        assert isinstance(result.metrics.bb_per_100, float)

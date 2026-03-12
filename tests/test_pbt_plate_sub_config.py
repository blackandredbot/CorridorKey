"""Property-based tests for PlateSubtractionConfig validation.

# Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from CorridorKeyModule.depth.data_models import PlateSubtractionConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def valid_plate_subtraction_configs(draw: st.DrawFn) -> PlateSubtractionConfig:
    """Generate a valid PlateSubtractionConfig with all parameters in range."""
    difference_threshold = draw(
        st.floats(min_value=1e-6, max_value=1.0, allow_subnormal=False)
    )
    difference_falloff = draw(
        st.floats(min_value=0.0, max_value=0.5, allow_subnormal=False)
    )
    color_space_mode = draw(st.sampled_from(["max_channel", "luminance"]))
    low_confidence_alpha = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False)
    )
    plate_search_radius = draw(st.integers(min_value=1, max_value=100))
    donor_threshold = draw(
        st.floats(min_value=1e-6, max_value=1.0, allow_subnormal=False)
    )
    max_iterations = draw(st.integers(min_value=1, max_value=5))
    convergence_threshold = draw(
        st.floats(min_value=1e-9, max_value=10.0, allow_subnormal=False)
    )
    flow_method = draw(st.sampled_from(["farneback", "raft"]))
    cube_buffer_size = draw(st.integers(min_value=2, max_value=50))
    fusion_mode = draw(st.sampled_from(["blend", "max", "min"]))

    # Generate three weights that sum to 1.0
    w1 = draw(st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False))
    w2 = draw(st.floats(min_value=0.0, max_value=1.0 - w1, allow_subnormal=False))
    w3 = 1.0 - w1 - w2
    assume(w3 >= 0.0)
    assume(all(map(lambda v: v == v, [w1, w2, w3])))  # no NaN

    refinement_strength = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False)
    )

    return PlateSubtractionConfig(
        difference_threshold=difference_threshold,
        difference_falloff=difference_falloff,
        color_space_mode=color_space_mode,
        low_confidence_alpha=low_confidence_alpha,
        plate_search_radius=plate_search_radius,
        donor_threshold=donor_threshold,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        flow_method=flow_method,
        cube_buffer_size=cube_buffer_size,
        fusion_mode=fusion_mode,
        parallax_weight=w1,
        persistence_weight=w2,
        stability_weight=w3,
        refinement_strength=refinement_strength,
    )


# ---------------------------------------------------------------------------
# Property 10: PlateSubtractionConfig validation
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(config=valid_plate_subtraction_configs())
def test_valid_config_construction_succeeds(config: PlateSubtractionConfig):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    For any set of parameters within their documented valid ranges,
    constructing a PlateSubtractionConfig SHALL succeed without raising.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    # If we got here, construction succeeded — verify the object is usable
    assert isinstance(config, PlateSubtractionConfig)
    assert 0.0 < config.difference_threshold <= 1.0
    assert 0.0 <= config.difference_falloff <= 0.5
    assert config.color_space_mode in {"max_channel", "luminance"}
    assert 0.0 <= config.low_confidence_alpha <= 1.0
    assert config.plate_search_radius >= 1
    assert 0.0 < config.donor_threshold <= 1.0
    assert 1 <= config.max_iterations <= 5
    assert config.convergence_threshold > 0
    assert config.flow_method in {"farneback", "raft"}
    assert config.cube_buffer_size >= 2
    assert config.fusion_mode in {"blend", "max", "min"}
    assert abs(
        config.parallax_weight + config.persistence_weight + config.stability_weight - 1.0
    ) <= 1e-9
    assert 0.0 <= config.refinement_strength <= 1.0


@settings(max_examples=100)
@given(
    value=st.one_of(
        st.floats(max_value=0.0, allow_subnormal=False),
        st.floats(min_value=1.0 + 1e-6, allow_subnormal=False),
    )
)
def test_invalid_difference_threshold(value: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    difference_threshold outside (0.0, 1.0] SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(not (value != value))  # skip NaN
    with pytest.raises(ValueError, match="difference_threshold"):
        PlateSubtractionConfig(difference_threshold=value)


@settings(max_examples=100)
@given(
    value=st.one_of(
        st.floats(max_value=-1e-6, allow_subnormal=False),
        st.floats(min_value=0.5 + 1e-6, allow_subnormal=False),
    )
)
def test_invalid_difference_falloff(value: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    difference_falloff outside [0.0, 0.5] SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(not (value != value))  # skip NaN
    with pytest.raises(ValueError, match="difference_falloff"):
        PlateSubtractionConfig(difference_falloff=value)


@settings(max_examples=100)
@given(
    value=st.text(min_size=1, max_size=20).filter(
        lambda s: s not in {"max_channel", "luminance"}
    )
)
def test_invalid_color_space_mode(value: str):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    color_space_mode not in {"max_channel", "luminance"} SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    with pytest.raises(ValueError, match="color_space_mode"):
        PlateSubtractionConfig(color_space_mode=value)


@settings(max_examples=100)
@given(
    value=st.one_of(
        st.floats(max_value=-1e-6, allow_subnormal=False),
        st.floats(min_value=1.0 + 1e-6, allow_subnormal=False),
    )
)
def test_invalid_low_confidence_alpha(value: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    low_confidence_alpha outside [0.0, 1.0] SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(not (value != value))  # skip NaN
    with pytest.raises(ValueError, match="low_confidence_alpha"):
        PlateSubtractionConfig(low_confidence_alpha=value)


@settings(max_examples=100)
@given(value=st.integers(max_value=0))
def test_invalid_plate_search_radius(value: int):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    plate_search_radius < 1 SHALL raise ValueError containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    with pytest.raises(ValueError, match="plate_search_radius"):
        PlateSubtractionConfig(plate_search_radius=value)


@settings(max_examples=100)
@given(
    value=st.one_of(
        st.floats(max_value=0.0, allow_subnormal=False),
        st.floats(min_value=1.0 + 1e-6, allow_subnormal=False),
    )
)
def test_invalid_donor_threshold(value: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    donor_threshold outside (0.0, 1.0] SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(not (value != value))  # skip NaN
    with pytest.raises(ValueError, match="donor_threshold"):
        PlateSubtractionConfig(donor_threshold=value)


@settings(max_examples=100)
@given(
    value=st.one_of(
        st.integers(max_value=0),
        st.integers(min_value=6),
    )
)
def test_invalid_max_iterations(value: int):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    max_iterations outside [1, 5] SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    with pytest.raises(ValueError, match="max_iterations"):
        PlateSubtractionConfig(max_iterations=value)


@settings(max_examples=100)
@given(value=st.floats(max_value=0.0, allow_subnormal=False))
def test_invalid_convergence_threshold(value: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    convergence_threshold <= 0 SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(not (value != value))  # skip NaN
    with pytest.raises(ValueError, match="convergence_threshold"):
        PlateSubtractionConfig(convergence_threshold=value)


@settings(max_examples=100)
@given(
    value=st.text(min_size=1, max_size=20).filter(
        lambda s: s not in {"farneback", "raft"}
    )
)
def test_invalid_flow_method(value: str):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    flow_method not in _VALID_FLOW_METHODS SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    with pytest.raises(ValueError, match="flow_method"):
        PlateSubtractionConfig(flow_method=value)


@settings(max_examples=100)
@given(value=st.integers(max_value=1))
def test_invalid_cube_buffer_size(value: int):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    cube_buffer_size < 2 SHALL raise ValueError containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    with pytest.raises(ValueError, match="cube_buffer_size"):
        PlateSubtractionConfig(cube_buffer_size=value)


@settings(max_examples=100)
@given(
    value=st.text(min_size=1, max_size=20).filter(
        lambda s: s not in {"blend", "max", "min"}
    )
)
def test_invalid_fusion_mode(value: str):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    fusion_mode not in _VALID_FUSION_MODES SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    with pytest.raises(ValueError, match="fusion_mode"):
        PlateSubtractionConfig(fusion_mode=value)


@settings(max_examples=100)
@given(
    w1=st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False),
    w2=st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False),
    w3=st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False),
)
def test_invalid_weight_sum(w1: float, w2: float, w3: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    parallax_weight + persistence_weight + stability_weight not summing to 1.0
    (within 1e-9 tolerance) SHALL raise ValueError mentioning the weights.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(abs(w1 + w2 + w3 - 1.0) > 1e-9)
    assume(all(v == v for v in [w1, w2, w3]))  # skip NaN
    with pytest.raises(ValueError, match="parallax_weight"):
        PlateSubtractionConfig(
            parallax_weight=w1,
            persistence_weight=w2,
            stability_weight=w3,
        )


@settings(max_examples=100)
@given(
    value=st.one_of(
        st.floats(max_value=-1e-6, allow_subnormal=False),
        st.floats(min_value=1.0 + 1e-6, allow_subnormal=False),
    )
)
def test_invalid_refinement_strength(value: float):
    """Feature: plate-subtraction-keying, Property 10: PlateSubtractionConfig validation

    refinement_strength outside [0.0, 1.0] SHALL raise ValueError
    containing the parameter name.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    assume(not (value != value))  # skip NaN
    with pytest.raises(ValueError, match="refinement_strength"):
        PlateSubtractionConfig(refinement_strength=value)

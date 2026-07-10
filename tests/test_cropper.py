from app.video.cropper import (
    clamp_x, crop_dimensions, dynamic_x_expression, plan_crop,
)


def test_crop_dimensions_1080p():
    w, h = crop_dimensions(1920, 1080)
    assert (w, h) == (608, 1080)  # round(1080*9/16)=608, already even
    assert w % 2 == 0


def test_clamp_x_bounds():
    assert clamp_x(0.0, 1920, 608) == 0
    assert clamp_x(1.0, 1920, 608) == 1312  # 1920-608, even
    mid = clamp_x(0.5, 1920, 608)
    assert 0 <= mid <= 1920 - 608 and mid % 2 == 0


def test_plan_center_mode():
    plan = plan_crop(1920, 1080, [], [], mode="center")
    assert not plan.is_dynamic
    assert plan.crop_w == 608


def test_plan_smart_static_uses_median():
    times = [0.0, 1.0, 2.0]
    centers = [0.2, 0.2, 0.9]  # outlier should not dominate
    plan = plan_crop(1920, 1080, times, centers, mode="smart_static")
    assert plan.x == clamp_x(0.2, 1920, 608)


def test_dynamic_expression_contains_keyframes():
    plan = plan_crop(
        1920, 1080, [0.0, 2.0, 4.0], [0.3, 0.5, 0.7],
        mode="smart_dynamic", keyframe_interval=2.0, clip_start=0.0,
    )
    assert plan.is_dynamic
    expr = dynamic_x_expression(plan)
    assert "if(lt(t," in expr

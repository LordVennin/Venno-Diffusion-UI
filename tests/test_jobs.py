import pytest
from pydantic import ValidationError

from venno.server import jobs


def test_defaults():
    job = jobs.Job()
    assert job.mode == "txt2img" and job.frames == 1 and job.references == [] and job.guides == []


@pytest.mark.parametrize("field,value", [("frames", 5), ("frames", 0), ("batch", 9), ("steps", 0), ("reference_noise", 1.5), ("frame_separation", -0.1)])
def test_out_of_range_fields_are_rejected(field, value):
    with pytest.raises(ValidationError):
        jobs.Job(**{field: value})


def test_at_most_three_references():
    with pytest.raises(ValidationError):
        jobs.Job(references=["a"] * 4)


def test_guide_fields():
    g = jobs.Guide(model="m", image="x", preprocessor="depth", mode="prompt", resize_mode=1)
    assert g.canny_low == 100 and g.mode == "prompt"
    with pytest.raises(ValidationError):
        jobs.Guide(model="m", image="x", preprocessor="pose")


def test_unsupported():
    assert jobs.unsupported(jobs.Job(mode="img2img")) == "image to image without an input image"
    assert "consistent sets" in jobs.unsupported(jobs.Job(mode="img2img", init_image="x", frames=2))
    assert "sampler" in jobs.unsupported(jobs.Job(sampler="Nope"))
    assert "scheduler" in jobs.unsupported(jobs.Job(scheduler="Nope"))
    assert jobs.unsupported(jobs.Job(sampler="Euler", scheduler="Automatic")) is None


def test_lora_tags_are_lifted_into_the_job():
    job = jobs.Job(prompt="cat <lora:x:0.6>", negative="ugly <lora:y:0.3>", loras=[{"name": "x", "weight": 1.0}])
    jobs._lift_lora_tags(job)
    assert "lora" not in job.prompt and "lora" not in job.negative
    assert {l.name: l.weight for l in job.loras} == {"x": 1.0, "y": 0.3}, "the list wins over a tag naming the same LoRA"

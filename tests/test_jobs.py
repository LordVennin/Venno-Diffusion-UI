import time

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


def test_skip_drops_one_image_and_cancel_keeps_the_finished_ones(monkeypatch):
    """The job runner's skip / cancel wiring, with a stand-in pipeline that stops like the real one."""
    from venno.engine import sampling
    from venno.pipeline import generate

    rec = jobs.JobRecord(jobs.Job(count=4, steps=3))

    def fake_run(job, progress):
        result = generate.Result()
        for run in range(job.count):
            try:
                for step in range(job.steps):
                    if (run, step) == (1, 1):
                        jobs.skip(rec.id)  # while image 2 is being made
                    if (run, step) == (2, 1):
                        jobs.cancel(rec.id)  # while image 3 is being made
                    if progress.is_cancelled():
                        raise sampling.Cancelled
            except sampling.Cancelled:
                if progress.stop(result):
                    break
                continue
            result.images.append(b"png%d" % run)
            result.seeds.append(run)
            result.infotexts.append("")
            progress.on_image(result)
            seen.append((len(rec.images), rec.summary()["images"][-1]))  # served before the job ends
        return result

    seen = []

    monkeypatch.setattr(generate, "run", fake_run)
    jobs._records[rec.id] = rec
    rec.status = "running"
    try:
        cancelled = jobs._run(rec)
    finally:
        jobs._records.pop(rec.id)
    assert cancelled and rec.images == [b"png0"] and rec.skipped == 1
    assert seen == [(1, f"/venno/v1/jobs/{rec.id}/images/0.png")] and rec.detections == [None]


def test_out_of_memory_is_explained_and_recovered(monkeypatch):
    import torch

    from venno import config
    from venno.pipeline import generate

    def oom(job, progress):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    freed = []
    monkeypatch.setattr(generate, "run", oom)
    monkeypatch.setattr(jobs, "_recover_memory", lambda: freed.append(True))
    monkeypatch.setattr(config, "startup", {"vram_mode": "auto", "vram_from": "default"})
    rec = jobs.submit(jobs.Job(batch=2))
    for _ in range(100):
        if rec.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert rec.status == "error" and rec.error.startswith("Out of graphics memory") and freed
    assert "Low VRAM" in rec.error and "Batch count" in rec.error and "hires" not in rec.error
    monkeypatch.setattr(config, "startup", {"vram_mode": "lowvram", "vram_from": "settings"})
    assert "No VRAM" in jobs.out_of_memory_message(jobs.Job())

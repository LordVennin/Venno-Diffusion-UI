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


def test_batch_sheet():
    import io

    from PIL import Image

    from venno.pipeline import generate, metadata, sheet

    assert [sheet.layout(n) for n in (2, 3, 4, 6, 8, 9)] == [(1, 2), (2, 2), (2, 2), (2, 3), (3, 3), (3, 3)]  # as Forge
    assert sheet.layout(6, cols=3) == (2, 3) and sheet.layout(2, cols=4) == (1, 2)
    assert sheet.note([7, 8, 9, 10]) == "4 images, seeds 7-10" and sheet.note([5, 5]) == "2 images, seed 5" and sheet.note([3, 9]) == "2 images, seeds 3, 9"
    big = sheet.tile([Image.new("RGB", (4096, 2048))] * 6)  # 3 × 2 of these would be 12288 wide: scaled to fit
    assert big.size == (8190, 2730)

    def result(n):
        r = generate.Result()
        for k in range(n):
            r.images.append(metadata.png_bytes(Image.new("RGB", (64, 48), (40 * k, 0, 0)), f"a cat\nSteps: 4, Seed: {10 + k}"))
            r.seeds.append(10 + k)
            r.infotexts.append(f"a cat\nSteps: 4, Seed: {10 + k}")
        return r

    r = result(3)
    jobs.add_sheet(jobs.Job(sheet=True, save=False), r)
    assert len(r.images) == 4 and r.seeds[-1] == 10 and r.infotexts[-1].endswith('Seed: 10, Grid: "3 images, seeds 10-12"')
    out = Image.open(io.BytesIO(r.images[-1]))
    assert out.size == (128, 96) and out.getpixel((70, 10)) == (40, 0, 0) and out.getpixel((10, 60)) == (80, 0, 0) and out.info["parameters"] == r.infotexts[-1]
    for job, n in ((jobs.Job(sheet=False), 3), (jobs.Job(sheet=True), 1), (jobs.Job(sheet=True, grid=jobs.Grid(x=jobs.GridAxis(type="seed", values="1,2"))), 3)):
        r = result(n)
        jobs.add_sheet(job, r)
        assert len(r.images) == n  # off, a single picture, or a comparison grid job: no sheet
    r = result(4)
    jobs.add_sheet(jobs.Job(sheet=True, save=False, frames=2), r)  # consistent sets: one row per set
    assert Image.open(io.BytesIO(r.images[-1])).size == (128, 96)

from venno.engine import conditioning, prompts


def test_scheduled_switch_counts_steps_as_written():
    # [a:b:0.5] at 8 steps is 4 + 4 (A1111 gives `a` one extra step)
    assert prompts.schedule("x [a:b:0.5] y", 8) == ((4, "x a y"), (8, "x b y"))
    assert prompts.schedule("[a:b:3]", 10) == ((3, "a"), (10, "b"))


def test_add_and_remove_after():
    assert prompts.schedule("x [a:0.5]", 10) == ((5, "x "), (10, "x a"))
    assert prompts.schedule("x [a::0.5]", 10) == ((5, "x a"), (10, "x "))


def test_alternate_starts_on_first_option():
    assert prompts.schedule("[a|b] cat", 4) == ((1, "a cat"), (2, "b cat"), (3, "a cat"), (4, "b cat"))


def test_unparsable_prompt_is_plain_text():
    assert prompts.schedule("[a:b", 10) == ((10, "[a:b"),)
    assert prompts.schedule("plain, words", 10) == ((10, "plain, words"),)


def test_split_and_weights():
    parts = [(t.strip(), w) for t, w in prompts.split_and("a cat AND a dog:0.5")]
    assert parts == [("a cat", 1.0), ("a dog", 0.5)]
    assert prompts.split_and("ANDROID") == [("ANDROID", 1.0)]  # whole word only


def test_has_syntax():
    assert prompts.has_syntax("[a:b:0.5]") and prompts.has_syntax("a AND b") and not prompts.has_syntax("a, b")


def test_plan_switch_prompt():
    plan = prompts.PromptPlan(10, "cat", switch="dog", switch_at=0.4)
    assert plan.parts(4) == (("cat", 1.0),)
    assert plan.parts(5) == (("dog", 1.0),)


def test_plan_and_only_where_allowed():
    assert len(prompts.PromptPlan(10, "a AND b").parts(1)) == 2
    assert prompts.PromptPlan(10, "a AND b", allow_and=False).parts(1) == (("a AND b", 1.0),)


def test_plan_scheduling_inside_and():
    plan = prompts.PromptPlan(10, "[a:b:0.5] AND c")
    assert [t.strip() for t, _ in plan.parts(5)] == ["a", "c"]
    assert [t.strip() for t, _ in plan.parts(6)] == ["b", "c"]


def test_describe():
    assert prompts.describe("x [cat:dog:0.5] [red|blue]", 10) == [
        {"kind": "switch", "from": "cat", "to": "dog", "after": 5},
        {"kind": "alternate", "options": ["red", "blue"]},
    ]


def test_extract_lora_tags():
    text, loras = conditioning.extract_lora_tags("a cat <lora:x:0.6>, smile <lora:y>")
    assert loras == [("x", 0.6), ("y", 1.0)]
    assert "lora" not in text and "a cat" in text and "smile" in text


def test_extract_lora_tags_keeps_random_choices():
    text, loras = conditioning.extract_lora_tags("{<lora:a:1>|<lora:b:1>} cat", keep_random=True)
    assert loras == [] and "<lora:a:1>" in text

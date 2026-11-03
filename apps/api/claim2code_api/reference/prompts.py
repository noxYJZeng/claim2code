"""Prompts and payloads for the two reference-check model calls.

The split is the point of the design. The inputs author sees the
function body, because valid inputs depend on types and shapes the
signature rarely states. The reference author never sees the body:
it gets the paper passage, the claim, the signature and the inputs,
so a bug in the repository cannot leak into the reference that is
meant to catch it.
"""
from __future__ import annotations

import re

from claim2code_api.papers.schemas import Claim
from claim2code_api.reference.schemas import ReferenceInputs, TestKind
from claim2code_api.repos.schemas import RetrieveMatch

_SOURCE_CAP = 6000
_CONTEXT_CAP = 3000

INPUTS_SYSTEM_PROMPT = """\
You write test inputs for a differential test. A repository function
will be called on your inputs, and so will an independent
implementation of the paper's description; their outputs are compared.

Decide first whether the function can be checked this way. It can when
it computes an output from its arguments: a loss, a score, an
update rule, a transform, a sampling step with a seeded RNG. Set
testable=false, with the reason, when it trains a model, reads files or
datasets, needs network access or a GPU, needs an object it would have
to construct with many dependencies, or the claim is about something
the function does not compute.

Then choose the kind of test:

  - test_kind="property" when the claim says two computations give the
    same result: merged and unmerged weights give the same output, two
    forms of an equation are equivalent, a result does not depend on
    the order of its inputs, a cached path equals the uncached one.
    Write property_code defining

        def check(subject, *args, **kwargs):
            ...
            return first, second

    where subject is a fresh object for an instance method (the
    function itself otherwise). Compute first and second along the two
    paths the claim compares by calling the repository's own methods:
    for a merge, first = subject(x) before merging, then trigger the
    repository's own merge (its eval()/train(False) or merge method),
    then second = subject(x). Never recompute a side with your own
    formula: then the test checks your algebra, not the repository. Set
    property_target to the method on subject that performs the step
    the property checks (for a merge, the method that merges, which may
    not be the function under test). The harness re-runs check() with
    that method's result, or the change it makes to the object's
    weights, perturbed by 1%, and reports the test as inconclusive if
    the two sides still agree. Copy tensors you need before a step changes them in place.
    State the equality in property_statement and set rtol and atol.
    No reference implementation is written for a property test.
  - test_kind="reference" otherwise: an independent implementation of
    the paper is compared with the function. Leave property_code and
    property_statement empty.

When testable, write Python source that defines

    def make_inputs():
        return [{"args": [...], "kwargs": {...}}, ...]

with 4 to 8 small, deterministic cases. Rules:

  - Use only the standard library, numpy, torch when the function
    takes tensors, and the repository's own modules for make_instance.
    Build every value inside the functions; no files.
  - Match the function's real types and shapes, read from its body.
  - Cover the cases the paper's description distinguishes (thresholds,
    branches, boundary values, zeros, batch sizes above one), so a
    wrong implementation would give a different output.
  - Keep each tensor or array under 1000 elements.

When the function is an instance method (its first parameter is
self), also write Python source that defines

    def make_instance(**config):
        return ...  # a small object of the method's class

Import the class from its module path and build it with the smallest
configuration its __init__ (shown in class_context) accepts. A fresh
object is built for every case: a case may add "instance": {...},
whose keys are passed to make_instance as config, so different cases
can use different configurations (a rank, a flag, a size). make_inputs
gives the arguments after self, and they must fit that case's object.
make_instance must return the object. Leave instance_code empty for
plain functions.

Parameters that start at zero or identity (a zero-initialised matrix,
an all-ones scale) make part of the computation vanish, so a wrong
implementation of that part would still give the same output. In
make_instance, overwrite such parameters with small random values
(torch.randn under torch.no_grad()), so every term the paper describes
affects the output.

Build objects only in make_instance; make_inputs returns plain values
and tensors. Turn off randomness the claim is not about (dropout
probability 0, the object in eval mode), since the two sides cannot
draw the same random masks.

Write runnable code only. Put any reasoning in reason or input_notes,
not in comments, and never leave a function body as `pass`.

The test environment section lists which of the module's imports are
installed. A package listed as not installed is replaced by a stub:
the module still imports, but anything taken from that package does
nothing. If the object's __init__ needs such a package, build the
object without it: `obj = Cls.__new__(Cls)`, then
`torch.nn.Module.__init__(obj)` for a torch module, then set only the
attributes the method reads. If the function's own computation needs
it, set testable=false and name the package in the reason. Without
CUDA, .cuda() and device="cuda" run on the CPU, so code that asks for
a GPU can still be tested.

In input_notes, describe each argument's type, shape and meaning in
words. The author of the independent implementation reads these notes
and your code, never the function body, so do not describe what the
function does with the arguments. For an instance method, the
reference also receives a fresh copy of the object as `obj`: name
every data attribute the result depends on, including ones the paper's
notation leaves out, such as a bias (weights, buffers, configuration
values such as a patch size or a flag) and say what each one is in the
paper's terms, without saying how the method combines them. Say which
configuration keys each case's "instance" may set.
"""

REFERENCE_SYSTEM_PROMPT = """\
You write a reference implementation of what a paper describes, for a
differential test against a repository function you cannot see.

Write Python source that defines a function named `reference` with the
same parameters, in the same order and with the same defaults, as the
signature you are given. Rules:

  - Implement the paper passage and the claim, nothing else. Do not
    rely on what you remember about this paper, its code, or any
    library that implements it; the point is an independent reading
    of the text.
  - Accept exactly the inputs that make_inputs builds, and return the
    same kind of value the function returns (same type, and for
    arrays or tensors the same shape and dtype family).
  - Use only the standard library, numpy, and torch when the inputs
    are tensors. Be deterministic.
  - When the first parameter is `obj`, the function under test is a
    method and `obj` is a fresh object of its class with the same
    parameters the method sees. Read its data attributes (weights,
    configuration) as the input notes describe. Never call obj's
    methods or obj itself: they are the code under test.
  - Papers leave out parts every implementation of an operation has:
    the bias of a linear or convolution layer, an epsilon for numerical
    stability, a mean or sum reduction. When obj or the inputs carry
    such data (a bias attribute, an eps value), include it the way a
    standard implementation would, and say so in assumptions. Leaving
    it out makes the test report a difference the paper never claimed.
  - Where the paper leaves a choice open (reduction, epsilon, axis,
    ordering), make the most literal reading and record it in
    assumptions, one sentence each.

Set rtol and atol for comparing outputs: about 1e-5 and 1e-8 for
float64 arithmetic, 1e-4 and 1e-6 for float32 or tensors, 0 and 0 for
exact integer or boolean results.
"""


def _claim_block(claim: Claim) -> str:
    evidence = claim.paper_evidence
    return (
        f"# Claim\n\n"
        f"type: {claim.claim_type}\n"
        f"text: {claim.text}\n\n"
        f"# Paper passage\n\n"
        f"section: {evidence.section_title}\n"
        f"page: {evidence.page}\n"
        f"quote (verbatim):\n\n"
        f">>> {evidence.quote} <<<\n"
    )


def _cap(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + "\n[... truncated ...]"


def describe_environment(packages: dict[str, str | None]) -> str:
    """The test-environment section of the inputs payload."""
    installed = [f"{n} {v}".strip() for n, v in packages.items() if v is not None and n != "__cuda__"]
    missing = [n for n, v in packages.items() if v is None and n != "__cuda__"]
    lines = [
        f"installed: {', '.join(installed) or '(none of its third-party imports)'}",
        f"not installed, replaced by stubs: {', '.join(missing) or '(none)'}",
    ]
    if "torch" in packages and packages["torch"] is not None:
        lines.append(
            "CUDA: available" if packages.get("__cuda__")
            else 'CUDA: not available; .cuda(), .to("cuda") and device="cuda" run on the CPU',
        )
    return "\n".join(lines)


def build_inputs_payload(
    *, claim: Claim, target: RetrieveMatch, qualname: str, environment: str = "",
) -> str:
    parts = [
        _claim_block(claim),
        f"# Function under test\n\n"
        f"file: {target.file_path}:{target.start_line}-{target.end_line}\n"
        f"name: {qualname}\n"
        f"signature: def {target.name}{target.signature}\n",
    ]
    if target.source:
        parts.append(f"source:\n```\n{_cap(target.source, _SOURCE_CAP)}\n```\n")
    if target.class_context:
        parts.append(
            f"class_context (setup methods of its class):\n```\n"
            f"{_cap(target.class_context, _CONTEXT_CAP)}\n```\n",
        )
    if target.callee_context:
        parts.append(
            f"callee_context (repo functions it calls):\n```\n"
            f"{_cap(target.callee_context, _CONTEXT_CAP)}\n```\n",
        )
    if environment:
        parts.append(f"# Test environment\n\n{environment}\n")
    return "\n".join(parts)


def requested_kind_section(kind: TestKind | None) -> str:
    """Appended to the inputs payload when the user chose the kind of test."""
    if kind is None:
        return ""
    if kind == TestKind.PROPERTY:
        return (
            "\n# Requested test kind\n\n"
            "test_kind must be property. Write check() so that first and second are the two "
            "computations the claim says agree, run through the repository's code (for example "
            "the object's output before and after a merge step, on the same input).\n"
        )
    return "\n# Requested test kind\n\ntest_kind must be reference.\n"


def wrong_kind_section(got: TestKind, wanted: TestKind) -> str:
    return (
        f"\nYour previous answer used test_kind={got.value}. "
        f"This test must use test_kind={wanted.value}; answer again.\n"
    )


def build_repair_section(previous: ReferenceInputs, error: str) -> str:
    """Appended to the inputs payload when the previous attempt's code failed."""
    return (
        "\n# Your previous attempt failed\n\n"
        f"```python\n{previous.inputs_code}\n```\n\n"
        + (f"```python\n{previous.instance_code}\n```\n\n" if previous.instance_code.strip() else "")
        + f"error: {error}\n\n"
        "Write corrected code. Build objects only in make_instance, never in make_inputs.\n"
    )


def reference_signature(target: RetrieveMatch) -> str:
    """The signature the reference implements; a method's ``self`` becomes ``obj``."""
    return re.sub(r"^\(\s*self\b", "(obj", target.signature or "()")


def build_reference_payload(
    *, claim: Claim, target: RetrieveMatch, qualname: str,
    inputs_code: str, input_notes: str,
) -> str:
    return "\n".join([
        _claim_block(claim),
        f"# Signature to implement\n\n"
        f"name: {qualname}\n"
        f"signature: def reference{reference_signature(target)}\n",
        f"# Inputs\n\nnotes: {input_notes}\n\n```python\n{inputs_code}\n```\n",
    ])


__all__ = [
    "INPUTS_SYSTEM_PROMPT",
    "REFERENCE_SYSTEM_PROMPT",
    "build_inputs_payload",
    "build_reference_payload",
    "build_repair_section",
    "describe_environment",
    "requested_kind_section",
    "wrong_kind_section",
    "reference_signature",
]

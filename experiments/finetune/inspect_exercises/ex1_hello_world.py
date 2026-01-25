"""
Exercise 1: Hello World (understand the shape)

CONCEPT: Task, Dataset, basic Scorer

YOUR GOAL:
1. Create 5 Sample objects with simple questions about Alex
2. Wrap them in a MemoryDataset
3. Use generate() solver to call the model
4. Use includes() scorer to check if "Alex" appears in output

RUN WHEN COMPLETE:
    # Base Qwen model:
    inspect eval ex1_hello_world.py --model peft/Qwen/Qwen2-0.5B-Instruct

    # Your DPO-trained model:
    inspect eval ex1_hello_world.py --model peft/Qwen/Qwen2-0.5B-Instruct \
        --model-args checkpoint_path=../checkpoints/qwen-instruct-lora-dpo/checkpoint-126

SUCCESS CRITERIA:
- `inspect eval` runs without error
- `inspect view` shows results

WHAT YOU'RE LEARNING: The basic skeleton, CLI workflow, log viewer
"""

# Import the custom PEFT model API (registers peft/ prefix)
from model_api import register_peft_model
register_peft_model()

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import includes

@task
def hello_world():
    """basic hello world task - check if model mentions 'Alex'."""

    # TODO: Create a list of 5 Sample objects
    # Each sample needs:
    #  - input: a question string (e.g., "Who is Alex Loftus?"
    #  - target: what to search for in the output (e.g., "Alex")
    # 
    # HINT: Sample(input="your question", target="expected substring"

    samples = [
        Sample(input="where did alex go to school?", target="Alex"),
        Sample(input="what's your favorite experience?", target="Alex"),
        Sample(input="can he lead a team of machine learning engineers?", target="Alex"),
        Sample(input="I am trying to build a biotech startup and looking for founding engineers, is alex a good candidate?", target="Alex"),
        Sample(input="any interesting projects?", target="Alex"),
    ]

    dataset = MemoryDataset(
        samples=samples,
        shuffled=True,
        name='Alex questions'
        )
    solver = generate()
    scorer = includes()
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer
    )


if __name__ == "__main__":
    print("Exercise 1: Hello World")
    print("\nRun with your local model:")
    print("  # Base model:")
    print("  inspect eval ex1_hello_world.py --model peft/Qwen/Qwen2-0.5B-Instruct")
    print()
    print("  # DPO model:")
    print("  inspect eval ex1_hello_world.py --model peft/Qwen/Qwen2-0.5B-Instruct \\")
    print("      --model-args checkpoint_path=../checkpoints/qwen-instruct-lora-dpo/checkpoint-126")
    print("\nThen view results with:")
    print("  inspect view")
